import pandas as pd
from datasets import load_dataset
from transformers import (
    GPT2Config, GPT2LMHeadModel, Trainer,
    TrainingArguments, DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast
)

# Load data
df = pd.read_csv("data/raw_smiles.csv")
df.columns
df["smiles"].dropna().to_csv("data/smiles.txt", index=False, header=False)

tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="tokenizer/smiles_tokenizer.json",
    bos_token="<bos>",
    eos_token="<eos>",
    pad_token="<pad>",
    unk_token="<unk>"
)

dataset = load_dataset("text", data_files={"train": "data/smiles.txt"})

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=128,
    n_embd=256,
    n_layer=6,
    n_head=8
)

model = GPT2LMHeadModel(config)

args = TrainingArguments(
    output_dir="model/gpt_smiles",
    per_device_train_batch_size=64,
    num_train_epochs=20,
    learning_rate=3e-4,
    save_steps=1000,
    logging_steps=200,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)

trainer.train()
model.save_pretrained("model/gpt_smiles")
tokenizer.save_pretrained("model/gpt_smiles")
