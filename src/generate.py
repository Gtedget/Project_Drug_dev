
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
import pandas as pd
import torch
from rdkit import Chem

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

def clean_and_canonicalize(smiles):
    smiles = smiles.replace(" ", "")  # Remove spaces
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Chem.MolToSmiles(mol)
    return None

smiles_list = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
cleaned = [clean_and_canonicalize(s) for s in smiles_list]
valid_smiles = [s for s in cleaned if s]

pd.DataFrame({"smiles": valid_smiles}).to_csv("data/generated_smiles.csv", index=False)
print(f"Generated and cleaned {len(valid_smiles)} valid molecules saved.")
