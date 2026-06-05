import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_MAP = {
    ("train", "gpt"): "train.py",
    ("generate", "gpt"): "generate.py",
    ("train", "conditional"): "train_conditional_from_raw_smiles.py",
    ("generate", "conditional"): "generate_conditional_from_raw_smiles.py",
    ("train", "gat"): "train_gat_conditional_from_raw_smiles.py",
    ("generate", "gat"): "generate_gat_conditional_from_raw_smiles.py",
    ("train", "diffusion"): "train_diffusion_from_raw_smiles.py",
    ("generate", "diffusion"): "generate_diffusion_from_raw_smiles.py",
}


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run GPT or standalone GAT drug-generation workflows from one entry point.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--model", choices=["gpt", "conditional", "gat", "diffusion"], required=True)

    generate_parser = subparsers.add_parser("generate", help="Generate molecules")
    generate_parser.add_argument("--model", choices=["gpt", "conditional", "gat", "diffusion"], required=True)

    args, forwarded_args = parser.parse_known_args()
    return args, forwarded_args


def main() -> None:
    args, forwarded_args = parse_args()

    script_name = SCRIPT_MAP[(args.command, args.model)]
    script_path = Path(__file__).with_name(script_name)

    command = [sys.executable, str(script_path), *forwarded_args]

    print("Dispatching workflow:")
    print(f"  command = {args.command}")
    print(f"  model   = {args.model}")
    print(f"  script  = {script_path.name}")
    if forwarded_args:
        print(f"  args    = {' '.join(forwarded_args)}")

    completed = subprocess.run(command, check=False)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()