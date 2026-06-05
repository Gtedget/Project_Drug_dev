import argparse
from pathlib import Path

from runtime_bootstrap import bootstrap_runtime

bootstrap_runtime()

import pandas as pd
from datasets import load_dataset
from transformers import (
    GPT2Config, GPT2LMHeadModel, Trainer,
    TrainingArguments, DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast
)

from sequence_utils import (
    ensure_sequence_column,
    get_default_tokenizer_path,
    infer_sequence_representation,
    load_tokenizer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the GPT-based SMILES generator.")
    parser.add_argument("--data_csv", type=str, default="data/raw_smiles.csv")
    parser.add_argument("--smiles_column", type=str, default="smiles")
    parser.add_argument("--sequence_column", type=str, default=None)
    parser.add_argument("--text_path", type=str, default="data/smiles.txt")
    parser.add_argument("--output_dir", type=str, default="model/gpt_smiles")
    parser.add_argument("--representation", type=str, default="auto", choices=["auto", "smiles", "selfies"])
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=200)
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_head", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.data_csv)
    sequence_column = args.sequence_column or args.smiles_column
    representation = infer_sequence_representation(sequence_column, args.representation)
    df = ensure_sequence_column(df, source_smiles_column=args.smiles_column, sequence_column=sequence_column, representation=representation)
    if sequence_column not in df.columns:
        raise ValueError(f"Missing required sequence column '{sequence_column}'")

    text_path = Path(args.text_path)
    text_path.parent.mkdir(parents=True, exist_ok=True)
    df[sequence_column].dropna().to_csv(text_path, index=False, header=False)

    tokenizer = load_tokenizer(args.tokenizer_path or get_default_tokenizer_path(representation))

    dataset = load_dataset("text", data_files={"train": str(text_path)})

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=args.max_length)

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=args.max_length,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        sequence_representation=representation,
        sequence_column=sequence_column,
    )

    model = GPT2LMHeadModel(config)

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset["train"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )

    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
