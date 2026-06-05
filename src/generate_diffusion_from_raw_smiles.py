from __future__ import annotations

import argparse

from runtime_bootstrap import bootstrap_runtime

bootstrap_runtime()

import pandas as pd
import torch
from transformers import PreTrainedTokenizerFast

from diffusion_smiles_generator import DiscreteDiffusionSmilesGenerator
from sequence_utils import load_tokenizer, sequence_to_smiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate molecules with the discrete diffusion SMILES model.")
    parser.add_argument("--checkpoint_path", type=str, default="model/diffusion_smiles.pt")
    parser.add_argument("--data_csv", type=str, default="data/raw_smiles.csv")
    parser.add_argument("--target_name", type=str, default=None)
    parser.add_argument("--alogp", type=float, default=None)
    parser.add_argument("--num_return_sequences", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--output_csv", type=str, default="data/generated_diffusion_smiles.csv")
    parser.add_argument("--representation", type=str, default="auto", choices=["auto", "smiles", "selfies"])
    parser.add_argument("--tokenizer_path", type=str, default=None)
    return parser.parse_args()


def normalize_logp(value: float, checkpoint: dict) -> float:
    stats = checkpoint.get("conditioning_stats", {})
    mean = float(stats.get("logp_mean", 0.0))
    std = float(stats.get("logp_std", 1.0)) or 1.0
    return (value - mean) / std


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    data_config = checkpoint.get("data_config", {})

    df = pd.read_csv(args.data_csv)
    smiles_column = data_config.get("smiles_column", "smiles")
    protein_column = data_config.get("protein_column")
    logp_column = data_config.get("logp_column")
    representation = data_config.get("sequence_representation", "smiles")
    if args.representation != "auto":
        representation = args.representation
    df = df[df[smiles_column].notnull()].reset_index(drop=True)
    default_row = df.iloc[0]

    target_name = args.target_name
    if target_name is None:
        target_name = str(default_row[protein_column]) if protein_column and protein_column in df.columns else "UNK_TARGET"

    if protein_column and protein_column in df.columns:
        matches = df[df[protein_column].astype(str) == str(target_name)]
        if not matches.empty:
            default_row = matches.iloc[0]

    alogp = args.alogp
    if alogp is None:
        alogp = float(default_row[logp_column]) if logp_column and logp_column in df.columns else 0.0

    tokenizer = load_tokenizer(args.tokenizer_path or data_config.get("tokenizer_path", "tokenizer/smiles_tokenizer.json"))
    model = DiscreteDiffusionSmilesGenerator(smiles_tokenizer=tokenizer, **checkpoint.get("model_config", {}))
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    proteins = [str(target_name)] * args.num_return_sequences
    normalized_logp = normalize_logp(float(alogp), checkpoint)
    logp_values = [normalized_logp] * args.num_return_sequences

    with torch.no_grad():
        smiles_list = model.generate(
            proteins=proteins,
            logp_values=logp_values,
            max_length=args.max_length,
            temperature=args.temperature,
        )

    cleaned = [sequence_to_smiles(smiles, representation=representation) for smiles in smiles_list]
    valid_smiles = [smiles for smiles in cleaned if smiles]
    out_df = pd.DataFrame(
        {
            "smiles": valid_smiles,
            "target_name": [target_name] * len(valid_smiles),
            "alogp": [alogp] * len(valid_smiles),
            "condition_alogp": [alogp] * len(valid_smiles),
        }
    )
    out_df.to_csv(args.output_csv, index=False)
    print(
        f"Generated and cleaned {len(valid_smiles)} valid molecules "
        f"with diffusion checkpoint {args.checkpoint_path}, saved to {args.output_csv}",
    )


if __name__ == "__main__":
    main()