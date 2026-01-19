from rdkit import Chem
import pandas as pd

df = pd.read_csv("data/raw_smiles.csv")
df = df.drop_duplicates(subset="smiles")
df = df.dropna(subset=["smiles"])  # Remove missing SMILES
df["valid"] = df["smiles"].apply(lambda s: isinstance(s, str) and Chem.MolFromSmiles(s) is not None)
df_valid = df[df["valid"]].drop("valid", axis=1)
df_valid.to_csv("data/filtered_smiles.csv", index=False)
print(f"Valid, unique SMILES: {len(df_valid)}")
