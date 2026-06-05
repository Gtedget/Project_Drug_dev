from __future__ import annotations

import argparse
from pathlib import Path

from runtime_bootstrap import bootstrap_runtime

bootstrap_runtime()

import pandas as pd
import torch
from transformers import PreTrainedTokenizerFast

from conditional_generator import ConditionalMoleculeDataset
from diffusion_smiles_generator import (
    DiffusionMoleculeDataset,
    DiffusionTrainer,
    DiffusionTrainingConfig,
    DiscreteDiffusionSmilesGenerator,
)
from sequence_utils import ensure_sequence_column, get_default_tokenizer_path, infer_sequence_representation, load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a discrete diffusion SMILES generator.")
    parser.add_argument("--data_csv", type=str, default="data/raw_smiles.csv")
    parser.add_argument("--smiles_column", type=str, default="smiles")
    parser.add_argument("--source_smiles_column", type=str, default="smiles")
    parser.add_argument("--protein_column", type=str, default="target_name")
    parser.add_argument("--logp_column", type=str, default="alogp")
    parser.add_argument("--output_path", type=str, default="model/diffusion_smiles.pt")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--num_diffusion_steps", type=int, default=32)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--representation", type=str, default="auto", choices=["auto", "smiles", "selfies"])
    parser.add_argument("--tokenizer_path", type=str, default=None)
    return parser.parse_args()


def resolve_optional_column(df: pd.DataFrame, requested: str) -> str | None:
    return requested if requested in df.columns else None


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.data_csv)
    representation = infer_sequence_representation(args.smiles_column, args.representation)
    df = ensure_sequence_column(df, source_smiles_column=args.source_smiles_column, sequence_column=args.smiles_column, representation=representation)
    df = df[df[args.smiles_column].notnull()].reset_index(drop=True)

    protein_column = resolve_optional_column(df, args.protein_column)
    logp_column = resolve_optional_column(df, args.logp_column)

    tokenizer = load_tokenizer(args.tokenizer_path or get_default_tokenizer_path(representation))
    base_dataset = ConditionalMoleculeDataset(
        df=df,
        smiles_column=args.smiles_column,
        protein_column=protein_column,
        logp_column=logp_column,
        max_smiles_length=args.max_length,
        smiles_tokenizer=tokenizer,
    )
    dataset = DiffusionMoleculeDataset(base_dataset)

    model = DiscreteDiffusionSmilesGenerator(
        smiles_tokenizer=tokenizer,
        max_length=args.max_length,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        num_diffusion_steps=args.num_diffusion_steps,
    )
    trainer = DiffusionTrainer(
        model=model,
        dataset=dataset,
        config=DiffusionTrainingConfig(
            batch_size=args.batch_size,
            lr=args.learning_rate,
            num_epochs=args.num_epochs,
        ),
    )
    trainer.train()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": model.get_config(),
        "data_config": {
            "smiles_column": args.smiles_column,
            "source_smiles_column": args.source_smiles_column,
            "protein_column": protein_column,
            "logp_column": logp_column,
            "max_length": args.max_length,
            "sequence_representation": representation,
            "tokenizer_path": args.tokenizer_path or get_default_tokenizer_path(representation),
        },
        "conditioning_stats": {
            "logp_mean": dataset.logp_mean,
            "logp_std": dataset.logp_std,
        },
    }
    torch.save(checkpoint, output_path)
    print(f"Saved diffusion checkpoint to {output_path}")


if __name__ == "__main__":
    main()