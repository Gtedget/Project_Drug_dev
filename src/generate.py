import argparse

from runtime_bootstrap import bootstrap_runtime

bootstrap_runtime()

import pandas as pd
import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

from sequence_utils import sequence_to_smiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate molecules with the GPT-based SMILES model.")
    parser.add_argument("--model_dir", type=str, default="model/gpt_smiles")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--num_return_sequences", type=int, default=100)
    parser.add_argument("--output_csv", type=str, default="data/generated_smiles.csv")
    parser.add_argument("--representation", type=str, default="auto", choices=["auto", "smiles", "selfies"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model_dir)
    model = GPT2LMHeadModel.from_pretrained(args.model_dir)
    model.eval()
    representation = getattr(model.config, "sequence_representation", None)
    if args.representation != "auto":
        representation = args.representation
    if representation is None:
        representation = "smiles"

    inputs = tokenizer("<bos>", return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_length=args.max_length,
        do_sample=True,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        num_return_sequences=args.num_return_sequences
    )

    smiles_list = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
    cleaned = [sequence_to_smiles(s, representation=representation) for s in smiles_list]
    valid_smiles = [s for s in cleaned if s]

    pd.DataFrame({"smiles": valid_smiles}).to_csv(args.output_csv, index=False)
    print(f"Generated and cleaned {len(valid_smiles)} valid molecules saved to {args.output_csv}.")


if __name__ == "__main__":
    main()
