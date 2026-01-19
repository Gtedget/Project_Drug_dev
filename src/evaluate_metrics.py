import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

# Load filtered generated SMILES
filtered_path = "data/generated_smiles_filtered.csv"
train_path = "data/filtered_smiles.csv"
df = pd.read_csv(filtered_path)
train_smiles = set(pd.read_csv(train_path)["smiles"])

# Metrics
valid_count = len(df)
unique_count = df["smiles"].nunique()
novel_count = sum(s not in train_smiles for s in df["smiles"])

print(f"Validity: {valid_count}")
print(f"Uniqueness: {unique_count}")
print(f"Novelty: {novel_count}")

# Visualization: Save a grid image of sample molecules
sample_smiles = df["smiles"].sample(min(10, valid_count))
mols = [Chem.MolFromSmiles(s) for s in sample_smiles]
img = Draw.MolsToGridImage(mols, molsPerRow=5)
img.save("data/sample_molecules.png")
print("Saved sample molecule grid to data/sample_molecules.png")
