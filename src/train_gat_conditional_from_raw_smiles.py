import argparse
from pathlib import Path

from runtime_bootstrap import bootstrap_runtime

bootstrap_runtime()

import pandas as pd
import torch
from transformers import PreTrainedTokenizerFast

from conditional_hybrid_graph_generator import (
    ConditionalGNNSmilesGenerator,
    ConditionalHybridMoleculeDataset,
    GNNConditionalTrainer,
    TrainingConfig,
)
from sequence_utils import ensure_sequence_column, get_default_tokenizer_path, infer_sequence_representation, load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a conditional SMILES generator with a graph protein encoder such as GAT.",
    )
    parser.add_argument("--data_csv", type=str, default="data/raw_smiles.csv")
    parser.add_argument("--smiles_column", type=str, default="smiles")
    parser.add_argument("--structure_smiles_column", type=str, default="smiles")
    parser.add_argument("--protein_column", type=str, default="target_name")
    parser.add_argument("--binding_site_column", type=str, default="binding_site")
    parser.add_argument("--logp_column", type=str, default="alogp")
    parser.add_argument("--protein_encoder_type", choices=["gat", "gcn"], default="gat")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--max_smiles_length", type=int, default=128)
    parser.add_argument("--output_path", type=str, default="model/conditional_gat_encoder_decoder.pt")
    parser.add_argument("--representation", type=str, default="auto", choices=["auto", "smiles", "selfies"])
    parser.add_argument("--tokenizer_path", type=str, default=None)
    return parser.parse_args()


def resolve_optional_column(df: pd.DataFrame, requested: str) -> str | None:
    return requested if requested in df.columns else None


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.data_csv)
    representation = infer_sequence_representation(args.smiles_column, args.representation)
    df = ensure_sequence_column(df, source_smiles_column=args.structure_smiles_column, sequence_column=args.smiles_column, representation=representation)
    df = df[df[args.smiles_column].notnull()].reset_index(drop=True)

    protein_col = resolve_optional_column(df, args.protein_column)
    binding_site_col = resolve_optional_column(df, args.binding_site_column)
    logp_col = resolve_optional_column(df, args.logp_column)

    print("Using columns:")
    print(f"  SMILES : {args.smiles_column}")
    print(f"  Protein: {protein_col if protein_col is not None else '[none / dummy]'}")
    print(f"  Pocket : {binding_site_col if binding_site_col is not None else '[none / dummy]'}")
    print(f"  logP   : {logp_col if logp_col is not None else '[none / dummy]'}")
    print(f"  Encoder: {args.protein_encoder_type}")

    tokenizer = load_tokenizer(args.tokenizer_path or get_default_tokenizer_path(representation))

    dataset = ConditionalHybridMoleculeDataset(
        df=df,
        smiles_column=args.smiles_column,
        protein_column=protein_col,
        binding_site_column=binding_site_col,
        logp_column=logp_col,
        structure_smiles_column=args.structure_smiles_column,
        max_smiles_length=args.max_smiles_length,
        smiles_tokenizer=tokenizer,
    )

    config = TrainingConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.num_epochs,
        max_smiles_length=args.max_smiles_length,
    )

    model = ConditionalGNNSmilesGenerator(
        smiles_tokenizer=tokenizer,
        descriptor_input_dim=dataset.descriptor_dim,
        protein_encoder_type=args.protein_encoder_type,
    )
    trainer = GNNConditionalTrainer(model=model, dataset=dataset, config=config)
    trainer.train()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": model.get_config(),
        "data_config": {
            "smiles_column": args.smiles_column,
            "structure_smiles_column": args.structure_smiles_column,
            "protein_column": protein_col,
            "binding_site_column": binding_site_col,
            "logp_column": logp_col,
            "descriptor_names": list(dataset.descriptor_names),
            "max_smiles_length": args.max_smiles_length,
            "sequence_representation": representation,
            "tokenizer_path": args.tokenizer_path or get_default_tokenizer_path(representation),
        },
        "conditioning_stats": {
            "logp_mean": dataset.logp_mean,
            "logp_std": dataset.logp_std,
            "descriptor_mean": dataset.descriptor_mean.tolist(),
            "descriptor_std": dataset.descriptor_std.tolist(),
        },
    }
    torch.save(checkpoint, output_path)
    print(f"Saved conditional {args.protein_encoder_type.upper()} checkpoint to {output_path}")


if __name__ == "__main__":
    main()