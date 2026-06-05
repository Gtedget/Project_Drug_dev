from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from runtime_bootstrap import bootstrap_runtime

bootstrap_runtime()

import pandas as pd

from comparison_metrics import evaluate_generated_file, load_train_smiles


def append_metrics(
    rows: list[dict[str, float | int | str]],
    model_name: str,
    output_csv: str,
    requested_samples: int,
    train_smiles: set[str],
    train_time: float,
    generation_time: float,
    args: argparse.Namespace,
) -> None:
    metrics = evaluate_generated_file(
        generated_csv=output_csv,
        requested_samples=requested_samples,
        train_smiles=train_smiles,
        smiles_column="smiles",
    )
    metrics.update({
        "model": model_name,
        "train_time_seconds": train_time,
        "inference_time_seconds": generation_time,
        "generation_time_seconds": generation_time,
        "requested_samples_per_second": requested_samples / max(generation_time, 1e-8),
        "valid_samples_per_second": metrics["valid_count"] / max(generation_time, 1e-8),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
    })
    rows.append(metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a matched comparison between GPT, conditional, hybrid GAT, and diffusion SMILES generators.",
    )
    parser.add_argument("--data_csv", type=str, default="data/raw_smiles.csv")
    parser.add_argument("--smiles_column", type=str, default="smiles")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument("--num_return_sequences", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--results_csv", type=str, default="data/model_comparison_all.csv")
    return parser.parse_args()


def run_and_time(command: list[str]) -> float:
    start = time.perf_counter()
    env = dict(os.environ)
    env.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    completed = subprocess.run(command, check=False, env=env)
    elapsed = time.perf_counter() - start
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(command)}")
    return elapsed


def main() -> None:
    args = parse_args()

    data_csv = Path(args.data_csv)
    if not data_csv.exists():
        raise FileNotFoundError(f"Missing comparison dataset: {data_csv}")

    train_smiles = load_train_smiles(str(data_csv), smiles_column=args.smiles_column)
    launcher = Path(__file__).with_name("run_drug_generator.py")

    gpt_output_dir = "model/gpt_smiles_compare"
    gpt_output_csv = "data/generated_smiles_gpt_compare.csv"
    conditional_checkpoint = "model/conditional_encoder_decoder_compare.pt"
    conditional_output_csv = "data/generated_smiles_conditional_compare.csv"
    gat_checkpoint = "model/conditional_hybrid_compare.pt"
    gat_output_csv = "data/generated_smiles_hybrid_compare.csv"
    diffusion_checkpoint = "model/diffusion_smiles_compare.pt"
    diffusion_output_csv = "data/generated_smiles_diffusion_compare.csv"

    rows: list[dict[str, float | int | str]] = []

    gpt_train_time = run_and_time([
        sys.executable,
        str(launcher),
        "train",
        "--model",
        "gpt",
        "--data_csv",
        str(data_csv),
        "--smiles_column",
        args.smiles_column,
        "--output_dir",
        gpt_output_dir,
        "--text_path",
        "data/smiles_smoke.txt",
        "--max_length",
        str(args.max_length),
        "--batch_size",
        str(args.batch_size),
        "--num_train_epochs",
        str(args.epochs),
        "--logging_steps",
        "10",
        "--save_steps",
        "1000",
    ])

    gpt_generation_time = run_and_time([
        sys.executable,
        str(launcher),
        "generate",
        "--model",
        "gpt",
        "--model_dir",
        gpt_output_dir,
        "--max_length",
        str(args.max_length),
        "--temperature",
        str(args.temperature),
        "--num_return_sequences",
        str(args.num_return_sequences),
        "--output_csv",
        gpt_output_csv,
    ])

    append_metrics(rows, "gpt", gpt_output_csv, args.num_return_sequences, train_smiles, gpt_train_time, gpt_generation_time, args)

    conditional_train_time = run_and_time([
        sys.executable,
        str(launcher),
        "train",
        "--model",
        "conditional",
        "--data_csv",
        str(data_csv),
        "--smiles_column",
        args.smiles_column,
        "--protein_column",
        "target_name",
        "--logp_column",
        "alogp",
        "--output_path",
        conditional_checkpoint,
        "--max_length",
        str(args.max_length),
        "--batch_size",
        str(args.batch_size),
        "--num_epochs",
        str(args.epochs),
        "--learning_rate",
        "3e-4",
    ])

    conditional_generation_time = run_and_time([
        sys.executable,
        str(launcher),
        "generate",
        "--model",
        "conditional",
        "--checkpoint_path",
        conditional_checkpoint,
        "--data_csv",
        str(data_csv),
        "--num_return_sequences",
        str(args.num_return_sequences),
        "--max_length",
        str(args.max_length),
        "--temperature",
        str(args.temperature),
        "--output_csv",
        conditional_output_csv,
    ])

    append_metrics(
        rows,
        "conditional_encoder_decoder",
        conditional_output_csv,
        args.num_return_sequences,
        train_smiles,
        conditional_train_time,
        conditional_generation_time,
        args,
    )

    gat_train_time = run_and_time([
        sys.executable,
        str(launcher),
        "train",
        "--model",
        "gat",
        "--data_csv",
        str(data_csv),
        "--smiles_column",
        args.smiles_column,
        "--protein_column",
        "target_name",
        "--binding_site_column",
        "binding_site",
        "--logp_column",
        "alogp",
        "--protein_encoder_type",
        "gat",
        "--batch_size",
        str(args.batch_size),
        "--num_epochs",
        str(args.epochs),
        "--max_smiles_length",
        str(args.max_length),
        "--output_path",
        gat_checkpoint,
    ])

    gat_generation_time = run_and_time([
        sys.executable,
        str(launcher),
        "generate",
        "--model",
        "gat",
        "--checkpoint_path",
        gat_checkpoint,
        "--data_csv",
        str(data_csv),
        "--num_return_sequences",
        str(args.num_return_sequences),
        "--max_length",
        str(args.max_length),
        "--temperature",
        str(args.temperature),
        "--output_csv",
        gat_output_csv,
    ])

    append_metrics(rows, "hybrid_gat", gat_output_csv, args.num_return_sequences, train_smiles, gat_train_time, gat_generation_time, args)

    diffusion_train_time = run_and_time([
        sys.executable,
        str(launcher),
        "train",
        "--model",
        "diffusion",
        "--data_csv",
        str(data_csv),
        "--smiles_column",
        args.smiles_column,
        "--protein_column",
        "target_name",
        "--logp_column",
        "alogp",
        "--output_path",
        diffusion_checkpoint,
        "--max_length",
        str(args.max_length),
        "--batch_size",
        str(args.batch_size),
        "--num_epochs",
        str(args.epochs),
        "--num_diffusion_steps",
        "24",
    ])

    diffusion_generation_time = run_and_time([
        sys.executable,
        str(launcher),
        "generate",
        "--model",
        "diffusion",
        "--checkpoint_path",
        diffusion_checkpoint,
        "--data_csv",
        str(data_csv),
        "--num_return_sequences",
        str(args.num_return_sequences),
        "--max_length",
        str(args.max_length),
        "--temperature",
        str(args.temperature),
        "--output_csv",
        diffusion_output_csv,
    ])

    append_metrics(
        rows,
        "diffusion",
        diffusion_output_csv,
        args.num_return_sequences,
        train_smiles,
        diffusion_train_time,
        diffusion_generation_time,
        args,
    )

    results_df = pd.DataFrame(rows)
    results_df.to_csv(args.results_csv, index=False)

    print("Comparison complete:")
    print(results_df.to_string(index=False))
    print(f"Saved comparison table to {args.results_csv}")


if __name__ == "__main__":
    main()