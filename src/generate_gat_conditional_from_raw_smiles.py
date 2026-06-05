import argparse

from runtime_bootstrap import bootstrap_runtime

bootstrap_runtime()

import pandas as pd
import torch
from transformers import PreTrainedTokenizerFast

from conditional_hybrid_graph_generator import ConditionalGNNSmilesGenerator, build_descriptor_vector
from sequence_utils import load_tokenizer, sequence_to_smiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate conditional SMILES from a trained GAT/GNN checkpoint.",
    )
    parser.add_argument("--checkpoint_path", type=str, default="model/conditional_gat_encoder_decoder.pt")
    parser.add_argument("--data_csv", type=str, default="data/raw_smiles.csv")
    parser.add_argument("--protein_value", type=str, default=None)
    parser.add_argument("--target_name", type=str, default=None)
    parser.add_argument("--binding_site", type=str, default=None)
    parser.add_argument("--seed_smiles", type=str, default=None)
    parser.add_argument("--logp_value", type=float, default=None)
    parser.add_argument("--alogp", type=float, default=None)
    parser.add_argument("--num_return_sequences", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--output_csv", type=str, default="data/generated_gat_conditional_smiles.csv")
    parser.add_argument("--representation", type=str, default="auto", choices=["auto", "smiles", "selfies"])
    parser.add_argument("--tokenizer_path", type=str, default=None)
    return parser.parse_args()


def load_checkpoint(path: str, tokenizer: PreTrainedTokenizerFast, device: torch.device) -> tuple[ConditionalGNNSmilesGenerator, dict]:
    checkpoint = torch.load(path, map_location=device)
    model_config = checkpoint.get("model_config", {})
    model = ConditionalGNNSmilesGenerator(smiles_tokenizer=tokenizer, **model_config)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, checkpoint


def normalize_logp(value: float, checkpoint: dict) -> float:
    stats = checkpoint.get("conditioning_stats", {})
    mean = float(stats.get("logp_mean", 0.0))
    std = float(stats.get("logp_std", 1.0)) or 1.0
    return (value - mean) / std


def normalize_descriptor_vector(values: list[float], checkpoint: dict) -> list[float]:
    stats = checkpoint.get("conditioning_stats", {})
    mean = stats.get("descriptor_mean", [0.0] * len(values))
    std = stats.get("descriptor_std", [1.0] * len(values))
    normalized = []
    for value, value_mean, value_std in zip(values, mean, std):
        denom = float(value_std) if float(value_std) != 0 else 1.0
        normalized.append((float(value) - float(value_mean)) / denom)
    return normalized


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.data_csv)
    df = df[df["smiles"].notnull()].reset_index(drop=True)
    checkpoint_data = torch.load(args.checkpoint_path, map_location="cpu")
    data_config = checkpoint_data.get("data_config", {})

    protein_column = data_config.get("protein_column")
    binding_site_column = data_config.get("binding_site_column")
    logp_column = data_config.get("logp_column")
    descriptor_names = data_config.get("descriptor_names", [])
    structure_smiles_column = data_config.get("structure_smiles_column", "smiles")
    representation = data_config.get("sequence_representation", "smiles")
    if args.representation != "auto":
        representation = args.representation

    default_row = df.iloc[0]
    protein_value = args.protein_value or args.target_name
    if protein_value is None:
        protein_value = str(default_row[protein_column]) if protein_column and protein_column in df.columns else "UNK_TARGET"

    if protein_column and protein_column in df.columns:
        matching_rows = df[df[protein_column].astype(str) == str(protein_value)]
        if not matching_rows.empty:
            default_row = matching_rows.iloc[0]

    binding_site_value = args.binding_site
    if binding_site_value is None:
        binding_site_value = (
            str(default_row[binding_site_column])
            if binding_site_column and binding_site_column in df.columns and not pd.isna(default_row[binding_site_column])
            else ""
        )

    seed_smiles_value = args.seed_smiles or str(default_row[structure_smiles_column])

    logp_value = args.logp_value if args.logp_value is not None else args.alogp
    if logp_value is None:
        logp_value = float(default_row[logp_column]) if logp_column and logp_column in df.columns else 0.0

    print("Using conditional context:")
    print(f"  protein = {protein_value}")
    print(f"  pocket  = {binding_site_value}")
    print(f"  seed    = {seed_smiles_value}")
    print(f"  logP    = {logp_value}")

    normalized_logp = normalize_logp(float(logp_value), checkpoint_data)
    descriptor_vector = build_descriptor_vector(descriptor_names, smiles=seed_smiles_value, row=default_row)
    normalized_descriptor_vector = normalize_descriptor_vector(descriptor_vector, checkpoint_data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(args.tokenizer_path or data_config.get("tokenizer_path", "tokenizer/smiles_tokenizer.json"))
    model, _ = load_checkpoint(args.checkpoint_path, tokenizer=tokenizer, device=device)

    proteins = [str(protein_value)] * args.num_return_sequences
    binding_sites = [str(binding_site_value)] * args.num_return_sequences
    seed_smiles = [str(seed_smiles_value)] * args.num_return_sequences
    logp_values = [normalized_logp] * args.num_return_sequences
    descriptor_vectors = [normalized_descriptor_vector] * args.num_return_sequences

    with torch.no_grad():
        batched_smiles = model.generate(
            proteins=proteins,
            binding_sites=binding_sites,
            logp_values=logp_values,
            descriptor_vectors=descriptor_vectors,
            seed_smiles=seed_smiles,
            max_length=args.max_length,
            num_samples=1,
            temperature=args.temperature,
        )

    smiles_list = [samples[0] for samples in batched_smiles if samples]
    cleaned = [sequence_to_smiles(smiles, representation=representation) for smiles in smiles_list]
    valid_smiles = [smiles for smiles in cleaned if smiles]

    output_rows = {
        "smiles": valid_smiles,
        "protein": [protein_value] * len(valid_smiles),
        "binding_site": [binding_site_value] * len(valid_smiles),
        "seed_smiles": [seed_smiles_value] * len(valid_smiles),
        "alogp": [logp_value] * len(valid_smiles),
        "condition_alogp": [logp_value] * len(valid_smiles),
    }
    for descriptor_name, descriptor_value in zip(descriptor_names, descriptor_vector):
        output_rows[f"condition_{descriptor_name}"] = [descriptor_value] * len(valid_smiles)

    out_df = pd.DataFrame(output_rows)
    out_df.to_csv(args.output_csv, index=False)

    print(
        f"Generated and cleaned {len(valid_smiles)} valid molecules "
        f"with checkpoint {args.checkpoint_path}, saved to {args.output_csv}",
    )


if __name__ == "__main__":
    main()