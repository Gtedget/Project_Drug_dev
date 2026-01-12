from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
import pandas as pd
import torch

tokenizer = PreTrainedTokenizerFast.from_pretrained("model/gpt_smiles")
model = GPT2LMHeadModel.from_pretrained("model/gpt_smiles")
model.eval()

inputs = tokenizer("<bos>", return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_length=128,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.8,
    num_return_sequences=100
)

smiles = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
pd.DataFrame({"smiles": smiles}).to_csv("data/generated_smiles.csv", index=False)
print("Generated molecules saved.")
