import argparse

from runtime_bootstrap import bootstrap_runtime

bootstrap_runtime()

import pandas as pd
import torch

from transformers import PreTrainedTokenizerFast

from conditional_generator import ConditionalMoleculeGenerator
from sequence_utils import load_tokenizer, sequence_to_smiles


def load_model(tokenizer: PreTrainedTokenizerFast, checkpoint_path: str, device: torch.device) -> tuple[ConditionalMoleculeGenerator, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint.get("model_config", {})
    condition_mode = model_config.get("condition_mode", "char_gru")
    target_vocab = model_config.get("target_vocab")

    model = ConditionalMoleculeGenerator(
        smiles_tokenizer=tokenizer,
        condition_mode=condition_mode,
        target_vocab=target_vocab,
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate conditional SMILES from a trained encoder-decoder using raw_smiles.csv",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="model/conditional_encoder_decoder.pt",
        help="Path to a trained conditional encoder-decoder checkpoint",
    )
    parser.add_argument(
        "--data_csv",
        type=str,
        default="data/raw_smiles.csv",
        help="Dataset used to resolve default conditioning values",
    )
    parser.add_argument(
        "--target_name",
        type=str,
        default=None,
        help="Target name to condition on (defaults to first target_name in raw_smiles.csv if available)",
    )
    parser.add_argument(
        "--alogp",
        type=float,
        default=None,
        help="logP (alogp) value to condition on (defaults to alogp of first row)",
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=100,
        help="Number of molecules to generate for the specified condition",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=64,
        help="Maximum SMILES length during generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=32,
        help="Restrict sampling to the top-k tokens at each decode step",
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=8,
        help="Minimum decode length before EOS is allowed",
    )
    parser.add_argument(
        "--eos_token_bonus",
        type=float,
        default=2.0,
        help="Logit bonus applied to EOS after the minimum length is reached",
    )
    parser.add_argument(
        "--max_repeat_tokens",
        type=int,
        default=3,
        help="Block extending runs of the same token beyond this length",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="data/generated_conditional_smiles.csv",
        help="Path to save generated, RDKit-cleaned SMILES",
    )
    parser.add_argument(
        "--representation",
        type=str,
        default="auto",
        choices=["auto", "smiles", "selfies"],
        help="Override the stored training representation for decoding",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Override the tokenizer path stored in the checkpoint",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint_data = torch.load(args.checkpoint_path, map_location="cpu")
    data_config = checkpoint_data.get("data_config", {})
    smiles_column = data_config.get("smiles_column", "smiles")
    protein_column = data_config.get("protein_column")
    logp_column = data_config.get("logp_column")
    representation = data_config.get("sequence_representation", "smiles")
    if args.representation != "auto":
        representation = args.representation
    tokenizer_path = args.tokenizer_path or data_config.get("tokenizer_path", "tokenizer/smiles_tokenizer.json")

    df = pd.read_csv(args.data_csv)
    df = df[df[smiles_column].notnull()].reset_index(drop=True)

    # Fallbacks from the data if the user did not specify conditions
    default_row = df.iloc[0]
    target_name = args.target_name
    if target_name is None:
        target_name = str(default_row[protein_column]) if protein_column and protein_column in df.columns else "UNK_TARGET"

    if protein_column and protein_column in df.columns:
        matching_rows = df[df[protein_column].astype(str) == str(target_name)]
        if not matching_rows.empty:
            default_row = matching_rows.iloc[0]

    alogp = args.alogp
    if alogp is None:
        if logp_column and logp_column in df.columns:
            alogp = float(default_row[logp_column])
        else:
            alogp = 0.0

    print("Using conditional context:")
    print(f"  target_name = {target_name}")
    print(f"  alogp       = {alogp}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(tokenizer_path)
    model, checkpoint = load_model(tokenizer, checkpoint_path=args.checkpoint_path, device=device)

    stats = checkpoint.get("conditioning_stats", {})
    logp_mean = float(stats.get("logp_mean", 0.0))
    logp_std = float(stats.get("logp_std", 1.0)) or 1.0
    normalized_alogp = (float(alogp) - logp_mean) / logp_std

    proteins = [str(target_name)] * args.num_return_sequences
    logp_values = [normalized_alogp] * args.num_return_sequences

    with torch.no_grad():
        smiles_list = model.generate(
            proteins=proteins,
            logp_values=logp_values,
            max_length=args.max_length,
            num_beams=1,
            temperature=args.temperature,
            top_k=args.top_k,
            min_length=args.min_length,
            eos_token_bonus=args.eos_token_bonus,
            max_repeat_tokens=args.max_repeat_tokens,
        )

    cleaned = [sequence_to_smiles(s, representation=representation) for s in smiles_list]
    valid_smiles = [s for s in cleaned if s]

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
        f"conditioned on target_name='{target_name}', alogp={alogp}, saved to {args.output_csv}",
    )


if __name__ == "__main__":
    main()
