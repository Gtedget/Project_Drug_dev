from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Install dependencies, prepare the target-aware SELFIES dataset, and run longer GPU training, "
            "generation, and evaluation for GPT, conditional, GAT, and diffusion models."
        )
    )
    parser.add_argument("--python_executable", type=str, default=sys.executable)
    parser.add_argument("--install", action="store_true", help="Install Python dependencies before running the pipeline.")
    parser.add_argument(
        "--torch_index_url",
        type=str,
        default=None,
        help="Optional PyTorch wheel index URL for a GPU build, for example https://download.pytorch.org/whl/cu124",
    )
    parser.add_argument("--skip_data_prep", action="store_true", help="Skip acquisition, curation, and tokenizer bootstrap.")
    parser.add_argument("--skip_train", action="store_true", help="Skip training and reuse existing checkpoints.")
    parser.add_argument("--skip_generate", action="store_true", help="Skip generation and reuse existing output CSV files.")
    parser.add_argument("--skip_evaluate", action="store_true", help="Skip RDKit evaluation and comparison CSV generation.")
    parser.add_argument("--force_acquisition", action="store_true", help="Rebuild the target activity dataset even if it already exists.")
    parser.add_argument("--force_curation", action="store_true", help="Rebuild the curated training CSV even if it already exists.")
    parser.add_argument("--force_tokenizer", action="store_true", help="Retrain the tokenizer even if it already exists.")
    parser.add_argument(
        "--force_train_existing",
        action="store_true",
        help="Retrain models even when the target checkpoint path already exists.",
    )
    parser.add_argument(
        "--force_generate_existing",
        action="store_true",
        help="Regenerate output CSV files even when they already exist.",
    )
    parser.add_argument("--run_name", type=str, default="vollab_long")
    parser.add_argument("--raw_input_csv", type=str, default="data/raw_smiles.csv")
    parser.add_argument("--activity_csv", type=str, default="data/target_activity_training.csv")
    parser.add_argument("--target_metadata_csv", type=str, default="data/target_metadata_enriched.csv")
    parser.add_argument("--curated_csv", type=str, default="data/target_activity_training_curated.csv")
    parser.add_argument("--curation_report_csv", type=str, default="data/target_activity_training_curation_report.csv")
    parser.add_argument("--failed_targets_csv", type=str, default="data/target_activity_failed_targets.csv")
    parser.add_argument("--requirements_path", type=str, default="requirements.txt")
    parser.add_argument("--representation", choices=["smiles", "selfies"], default="selfies")
    parser.add_argument("--sequence_column", type=str, default="selfies")
    parser.add_argument("--source_smiles_column", type=str, default="smiles")
    parser.add_argument("--target_column", type=str, default="target_chembl_id")
    parser.add_argument("--target_name_column", type=str, default="target_name")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer/selfies_tokenizer.json")
    parser.add_argument("--tokenizer_text_path", type=str, default="data/selfies_activity_vollab.txt")
    parser.add_argument("--tokenizer_vocab_size", type=int, default=512)
    parser.add_argument("--max_length", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--gpt_batch_size", type=int, default=32)
    parser.add_argument("--conditional_batch_size", type=int, default=32)
    parser.add_argument("--gat_batch_size", type=int, default=16)
    parser.add_argument("--diffusion_batch_size", type=int, default=32)
    parser.add_argument("--num_return_sequences", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--gpt_save_steps", type=int, default=500)
    parser.add_argument("--gpt_logging_steps", type=int, default=50)
    parser.add_argument("--gpt_n_embd", type=int, default=256)
    parser.add_argument("--gpt_n_layer", type=int, default=6)
    parser.add_argument("--gpt_n_head", type=int, default=8)
    parser.add_argument("--diffusion_steps", type=int, default=32)
    parser.add_argument("--max_targets_per_smiles", type=int, default=2)
    parser.add_argument("--max_activities_per_target", type=int, default=100)
    parser.add_argument("--results_csv", type=str, default=None)
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def path_arg(path: Path) -> str:
    return str(path.resolve())


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run_command(command: list[str], *, env_overrides: dict[str, str] | None = None) -> float:
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    if env_overrides:
        env.update(env_overrides)

    print(f"\n>>> {shell_join(command)}")
    start = time.perf_counter()
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=False)
    elapsed = time.perf_counter() - start
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {shell_join(command)}")
    print(f"Completed in {elapsed:.1f}s")
    return elapsed


def install_dependencies(args: argparse.Namespace) -> None:
    requirements_path = resolve_path(args.requirements_path)
    if not requirements_path.exists():
        raise FileNotFoundError(f"Missing requirements file: {requirements_path}")

    run_command([args.python_executable, "-m", "pip", "install", "--upgrade", "pip"])
    run_command([args.python_executable, "-m", "pip", "install", "-r", path_arg(requirements_path)])
    if args.torch_index_url:
        run_command(
            [
                args.python_executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "torch",
                "--index-url",
                args.torch_index_url,
            ]
        )


def read_csv_header(path: Path) -> list[str]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        return next(reader)


def read_first_row(path: Path) -> dict[str, str]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            return {key: (value or "") for key, value in row.items()}
    raise ValueError(f"CSV has no data rows: {path}")


def ensure_target_activity_dataset(args: argparse.Namespace) -> None:
    raw_input_csv = resolve_path(args.raw_input_csv)
    activity_csv = resolve_path(args.activity_csv)
    target_metadata_csv = resolve_path(args.target_metadata_csv)
    failed_targets_csv = resolve_path(args.failed_targets_csv)

    if activity_csv.exists() and target_metadata_csv.exists() and not args.force_acquisition:
        return

    if not raw_input_csv.exists():
        raise FileNotFoundError(f"Missing raw input CSV: {raw_input_csv}")

    raw_header = set(read_csv_header(raw_input_csv))
    if args.target_column not in raw_header:
        raise ValueError(
            f"Cannot acquire target-aware data because '{args.target_column}' is missing from {raw_input_csv}. "
            "Provide an existing target activity CSV or a raw CSV with target identifiers."
        )

    run_command(
        [
            args.python_executable,
            path_arg(SRC_DIR / "build_target_activity_dataset.py"),
            "--input_csv",
            path_arg(raw_input_csv),
            "--output_csv",
            path_arg(activity_csv),
            "--target_metadata_csv",
            path_arg(target_metadata_csv),
            "--target_column",
            args.target_column,
            "--target_name_column",
            args.target_name_column,
            "--max_activities_per_target",
            str(args.max_activities_per_target),
            "--selfies_column",
            args.sequence_column,
            "--failed_targets_csv",
            path_arg(failed_targets_csv),
        ]
    )


def ensure_curated_dataset(args: argparse.Namespace) -> None:
    activity_csv = resolve_path(args.activity_csv)
    curated_csv = resolve_path(args.curated_csv)
    report_csv = resolve_path(args.curation_report_csv)
    if curated_csv.exists() and report_csv.exists() and not args.force_curation:
        return

    if not activity_csv.exists():
        raise FileNotFoundError(f"Missing activity CSV for curation: {activity_csv}")

    run_command(
        [
            args.python_executable,
            path_arg(SRC_DIR / "curate_training_data.py"),
            "--input_csv",
            path_arg(activity_csv),
            "--output_csv",
            path_arg(curated_csv),
            "--report_csv",
            path_arg(report_csv),
            "--smiles_column",
            args.source_smiles_column,
            "--target_column",
            args.target_column,
            "--max_smiles_length",
            str(args.max_length),
            "--max_targets_per_smiles",
            str(args.max_targets_per_smiles),
            "--selfies_column",
            args.sequence_column,
        ]
    )


def ensure_tokenizer(args: argparse.Namespace) -> None:
    tokenizer_path = resolve_path(args.tokenizer_path)
    if tokenizer_path.exists() and not args.force_tokenizer:
        return

    curated_csv = resolve_path(args.curated_csv)
    if not curated_csv.exists():
        raise FileNotFoundError(f"Missing curated CSV for tokenizer training: {curated_csv}")

    run_command(
        [
            args.python_executable,
            path_arg(SRC_DIR / "tokenizer_train.py"),
            "--data_csv",
            path_arg(curated_csv),
            "--sequence_column",
            args.sequence_column,
            "--source_smiles_column",
            args.source_smiles_column,
            "--representation",
            args.representation,
            "--text_path",
            path_arg(resolve_path(args.tokenizer_text_path)),
            "--tokenizer_output",
            path_arg(tokenizer_path),
            "--vocab_size",
            str(args.tokenizer_vocab_size),
        ]
    )


def prepare_data(args: argparse.Namespace) -> None:
    ensure_target_activity_dataset(args)
    ensure_curated_dataset(args)
    ensure_tokenizer(args)


def derive_generation_context(curated_csv: Path, target_column: str, source_smiles_column: str) -> dict[str, str]:
    first_row = read_first_row(curated_csv)
    target_value = first_row.get(target_column, "")
    if not target_value:
        raise ValueError(f"Curated dataset {curated_csv} does not contain a usable '{target_column}' value")
    seed_smiles = first_row.get(source_smiles_column, "")
    return {
        "target_value": target_value,
        "seed_smiles": seed_smiles,
    }


def build_run_paths(args: argparse.Namespace) -> dict[str, Path]:
    run_name = args.run_name
    return {
        "gpt_model": resolve_path(f"model/gpt_{run_name}"),
        "conditional_model": resolve_path(f"model/conditional_{run_name}.pt"),
        "gat_model": resolve_path(f"model/conditional_gat_{run_name}.pt"),
        "diffusion_model": resolve_path(f"model/diffusion_{run_name}.pt"),
        "gpt_output": resolve_path(f"data/generated_gpt_{run_name}.csv"),
        "conditional_output": resolve_path(f"data/generated_conditional_{run_name}.csv"),
        "gat_output": resolve_path(f"data/generated_gat_{run_name}.csv"),
        "diffusion_output": resolve_path(f"data/generated_diffusion_{run_name}.csv"),
        "results_csv": resolve_path(args.results_csv or f"data/model_comparison_{run_name}.csv"),
        "manifest_json": resolve_path(f"data/run_manifest_{run_name}.json"),
    }


def build_plan(args: argparse.Namespace, context: dict[str, str], paths: dict[str, Path]) -> list[dict[str, Any]]:
    curated_csv = resolve_path(args.curated_csv)
    tokenizer_path = resolve_path(args.tokenizer_path)
    text_path = resolve_path(args.tokenizer_text_path)

    common_sequence_args = [
        "--data_csv",
        path_arg(curated_csv),
        "--representation",
        args.representation,
        "--tokenizer_path",
        path_arg(tokenizer_path),
    ]

    return [
        {
            "name": "gpt",
            "checkpoint": path_arg(paths["gpt_model"]),
            "output_csv": path_arg(paths["gpt_output"]),
            "force_train": args.force_train_existing,
            "force_generate": args.force_generate_existing,
            "train_command": [
                args.python_executable,
                path_arg(SRC_DIR / "train.py"),
                *common_sequence_args,
                "--smiles_column",
                args.source_smiles_column,
                "--sequence_column",
                args.sequence_column,
                "--text_path",
                path_arg(text_path),
                "--output_dir",
                path_arg(paths["gpt_model"]),
                "--max_length",
                str(args.max_length),
                "--batch_size",
                str(args.gpt_batch_size),
                "--num_train_epochs",
                str(args.epochs),
                "--learning_rate",
                str(args.learning_rate),
                "--save_steps",
                str(args.gpt_save_steps),
                "--logging_steps",
                str(args.gpt_logging_steps),
                "--n_embd",
                str(args.gpt_n_embd),
                "--n_layer",
                str(args.gpt_n_layer),
                "--n_head",
                str(args.gpt_n_head),
            ],
            "generate_command": [
                args.python_executable,
                path_arg(SRC_DIR / "generate.py"),
                "--model_dir",
                path_arg(paths["gpt_model"]),
                "--representation",
                args.representation,
                "--max_length",
                str(args.max_length),
                "--temperature",
                str(args.temperature),
                "--num_return_sequences",
                str(args.num_return_sequences),
                "--output_csv",
                path_arg(paths["gpt_output"]),
            ],
        },
        {
            "name": "conditional",
            "checkpoint": path_arg(paths["conditional_model"]),
            "output_csv": path_arg(paths["conditional_output"]),
            "force_train": args.force_train_existing,
            "force_generate": args.force_generate_existing,
            "train_command": [
                args.python_executable,
                path_arg(SRC_DIR / "train_conditional_from_raw_smiles.py"),
                *common_sequence_args,
                "--smiles_column",
                args.sequence_column,
                "--source_smiles_column",
                args.source_smiles_column,
                "--protein_column",
                args.target_column,
                "--output_path",
                path_arg(paths["conditional_model"]),
                "--max_length",
                str(args.max_length),
                "--batch_size",
                str(args.conditional_batch_size),
                "--num_epochs",
                str(args.epochs),
                "--learning_rate",
                str(args.learning_rate),
                "--condition_mode",
                "target_lookup",
            ],
            "generate_command": [
                args.python_executable,
                path_arg(SRC_DIR / "generate_conditional_from_raw_smiles.py"),
                "--checkpoint_path",
                path_arg(paths["conditional_model"]),
                "--data_csv",
                path_arg(curated_csv),
                "--target_name",
                context["target_value"],
                "--representation",
                args.representation,
                "--tokenizer_path",
                path_arg(tokenizer_path),
                "--num_return_sequences",
                str(args.num_return_sequences),
                "--max_length",
                str(args.max_length),
                "--temperature",
                str(args.temperature),
                "--output_csv",
                path_arg(paths["conditional_output"]),
            ],
        },
        {
            "name": "gat",
            "checkpoint": path_arg(paths["gat_model"]),
            "output_csv": path_arg(paths["gat_output"]),
            "force_train": args.force_train_existing,
            "force_generate": args.force_generate_existing,
            "train_command": [
                args.python_executable,
                path_arg(SRC_DIR / "train_gat_conditional_from_raw_smiles.py"),
                *common_sequence_args,
                "--smiles_column",
                args.sequence_column,
                "--structure_smiles_column",
                args.source_smiles_column,
                "--protein_column",
                args.target_column,
                "--protein_encoder_type",
                "gat",
                "--batch_size",
                str(args.gat_batch_size),
                "--lr",
                str(args.learning_rate),
                "--num_epochs",
                str(args.epochs),
                "--max_smiles_length",
                str(args.max_length),
                "--output_path",
                path_arg(paths["gat_model"]),
            ],
            "generate_command": [
                args.python_executable,
                path_arg(SRC_DIR / "generate_gat_conditional_from_raw_smiles.py"),
                "--checkpoint_path",
                path_arg(paths["gat_model"]),
                "--data_csv",
                path_arg(curated_csv),
                "--protein_value",
                context["target_value"],
                "--seed_smiles",
                context["seed_smiles"],
                "--representation",
                args.representation,
                "--tokenizer_path",
                path_arg(tokenizer_path),
                "--num_return_sequences",
                str(args.num_return_sequences),
                "--max_length",
                str(args.max_length),
                "--temperature",
                str(args.temperature),
                "--output_csv",
                path_arg(paths["gat_output"]),
            ],
        },
        {
            "name": "diffusion",
            "checkpoint": path_arg(paths["diffusion_model"]),
            "output_csv": path_arg(paths["diffusion_output"]),
            "force_train": args.force_train_existing,
            "force_generate": args.force_generate_existing,
            "train_command": [
                args.python_executable,
                path_arg(SRC_DIR / "train_diffusion_from_raw_smiles.py"),
                *common_sequence_args,
                "--smiles_column",
                args.sequence_column,
                "--source_smiles_column",
                args.source_smiles_column,
                "--protein_column",
                args.target_column,
                "--output_path",
                path_arg(paths["diffusion_model"]),
                "--max_length",
                str(args.max_length),
                "--batch_size",
                str(args.diffusion_batch_size),
                "--num_epochs",
                str(args.epochs),
                "--learning_rate",
                str(args.learning_rate),
                "--num_diffusion_steps",
                str(args.diffusion_steps),
            ],
            "generate_command": [
                args.python_executable,
                path_arg(SRC_DIR / "generate_diffusion_from_raw_smiles.py"),
                "--checkpoint_path",
                path_arg(paths["diffusion_model"]),
                "--data_csv",
                path_arg(curated_csv),
                "--target_name",
                context["target_value"],
                "--representation",
                args.representation,
                "--tokenizer_path",
                path_arg(tokenizer_path),
                "--num_return_sequences",
                str(args.num_return_sequences),
                "--max_length",
                str(args.max_length),
                "--temperature",
                str(args.temperature),
                "--output_csv",
                path_arg(paths["diffusion_output"]),
            ],
        },
    ]


def ensure_parent_directories(paths: dict[str, Path]) -> None:
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)


def artifact_exists(path_str: str) -> bool:
    path = Path(path_str)
    if path.is_dir():
        return any(path.iterdir())
    return path.exists() and path.stat().st_size > 0


def train_models(plan: list[dict[str, Any]]) -> dict[str, float]:
    train_times: dict[str, float] = {}
    for entry in plan:
        checkpoint_exists = artifact_exists(entry["checkpoint"])
        if checkpoint_exists and not entry["force_train"]:
            print(f"Skipping training for {entry['name']}: found existing checkpoint at {entry['checkpoint']}")
            train_times[entry["name"]] = 0.0
            continue
        train_times[entry["name"]] = run_command(entry["train_command"])
    return train_times


def generate_outputs(plan: list[dict[str, Any]]) -> dict[str, float]:
    generation_times: dict[str, float] = {}
    for entry in plan:
        output_exists = artifact_exists(entry["output_csv"])
        if output_exists and not entry["force_generate"]:
            print(f"Skipping generation for {entry['name']}: found existing output at {entry['output_csv']}")
            generation_times[entry["name"]] = 0.0
            continue
        generation_times[entry["name"]] = run_command(entry["generate_command"])
    return generation_times


def evaluate_outputs(
    args: argparse.Namespace,
    plan: list[dict[str, Any]],
    paths: dict[str, Path],
    train_times: dict[str, float],
    generation_times: dict[str, float],
) -> list[dict[str, Any]]:
    sys.path.insert(0, str(SRC_DIR))
    from runtime_bootstrap import bootstrap_runtime

    bootstrap_runtime()

    import pandas as pd
    from comparison_metrics import evaluate_generated_file, load_train_smiles

    train_smiles = load_train_smiles(str(resolve_path(args.curated_csv)), smiles_column=args.source_smiles_column)
    rows: list[dict[str, Any]] = []
    for entry in plan:
        metrics = evaluate_generated_file(
            generated_csv=entry["output_csv"],
            requested_samples=args.num_return_sequences,
            train_smiles=train_smiles,
            smiles_column="smiles",
        )
        metrics.update(
            {
                "model": entry["name"],
                "checkpoint": entry["checkpoint"],
                "epochs": args.epochs,
                "max_length": args.max_length,
                "representation": args.representation,
                "train_time_seconds": train_times.get(entry["name"], 0.0),
                "generation_time_seconds": generation_times.get(entry["name"], 0.0),
            }
        )
        rows.append(metrics)

    results_df = pd.DataFrame(rows)
    results_df.to_csv(paths["results_csv"], index=False)
    print(f"Saved comparison table to {paths['results_csv']}")
    return rows


def save_manifest(args: argparse.Namespace, paths: dict[str, Path], plan: list[dict[str, Any]], metrics: list[dict[str, Any]]) -> None:
    manifest = {
        "run_name": args.run_name,
        "python_executable": args.python_executable,
        "representation": args.representation,
        "curated_csv": path_arg(resolve_path(args.curated_csv)),
        "tokenizer_path": path_arg(resolve_path(args.tokenizer_path)),
        "paths": {key: path_arg(value) for key, value in paths.items()},
        "models": [
            {
                "name": entry["name"],
                "checkpoint": entry["checkpoint"],
                "output_csv": entry["output_csv"],
            }
            for entry in plan
        ],
        "metrics": metrics,
    }
    paths["manifest_json"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved run manifest to {paths['manifest_json']}")


def main() -> None:
    args = parse_args()

    if args.install:
        install_dependencies(args)

    if not args.skip_data_prep:
        prepare_data(args)

    curated_csv = resolve_path(args.curated_csv)
    tokenizer_path = resolve_path(args.tokenizer_path)
    if not curated_csv.exists():
        raise FileNotFoundError(f"Missing curated dataset: {curated_csv}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Missing tokenizer: {tokenizer_path}")

    context = derive_generation_context(curated_csv, target_column=args.target_column, source_smiles_column=args.source_smiles_column)
    paths = build_run_paths(args)
    ensure_parent_directories(paths)
    plan = build_plan(args, context, paths)

    train_times = {entry["name"]: 0.0 for entry in plan}
    generation_times = {entry["name"]: 0.0 for entry in plan}

    if not args.skip_train:
        train_times = train_models(plan)

    if not args.skip_generate:
        generation_times = generate_outputs(plan)

    metrics: list[dict[str, Any]] = []
    if not args.skip_evaluate:
        metrics = evaluate_outputs(args, plan, paths, train_times, generation_times)

    save_manifest(args, paths, plan, metrics)
    print("Long GPU pipeline finished successfully.")


if __name__ == "__main__":
    main()