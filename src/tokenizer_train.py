import argparse
from pathlib import Path

from runtime_bootstrap import bootstrap_runtime

bootstrap_runtime()

import pandas as pd
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

from sequence_utils import (
    ensure_sequence_column,
    get_default_tokenizer_path,
    infer_sequence_representation,
    tokenizer_regex_for_representation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tokenizer for SMILES or SELFIES sequences.")
    parser.add_argument("--data_csv", type=str, default="data/raw_smiles.csv")
    parser.add_argument("--sequence_column", type=str, default="smiles")
    parser.add_argument("--source_smiles_column", type=str, default="smiles")
    parser.add_argument("--representation", type=str, default="auto", choices=["auto", "smiles", "selfies"])
    parser.add_argument("--text_path", type=str, default=None)
    parser.add_argument("--tokenizer_output", type=str, default=None)
    parser.add_argument("--vocab_size", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    representation = infer_sequence_representation(args.sequence_column, args.representation)
    df = pd.read_csv(args.data_csv)
    df = ensure_sequence_column(df, source_smiles_column=args.source_smiles_column, sequence_column=args.sequence_column, representation=representation)
    if args.sequence_column not in df.columns:
        raise ValueError(f"Missing required sequence column '{args.sequence_column}'")

    text_path = args.text_path or ("data/selfies.txt" if representation == "selfies" else "data/smiles.txt")
    tokenizer_output = args.tokenizer_output or get_default_tokenizer_path(representation)
    text_file = Path(text_path)
    text_file.parent.mkdir(parents=True, exist_ok=True)
    df[args.sequence_column].dropna().to_csv(text_file, index=False, header=False)

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Split(
        pattern=tokenizer_regex_for_representation(representation),
        behavior="isolated",
    )
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
    )
    tokenizer.train([str(text_file)], trainer)
    Path(tokenizer_output).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(tokenizer_output)
    print(f"Tokenizer saved to {tokenizer_output} for {representation} sequences.")


if __name__ == "__main__":
    main()
