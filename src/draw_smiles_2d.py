import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import os

# Load the generated SMILES
input_csv = os.path.join('data', 'generated_smiles_valid.csv')
df = pd.read_csv(input_csv)

# Ensure the column is named 'smiles' (adjust if needed)
if 'smiles' not in df.columns:
    df.columns = ['smiles']

# Output directory for images
output_dir = os.path.join('data', 'smiles_2d_images')
os.makedirs(output_dir, exist_ok=True)

for idx, row in df.iterrows():
    smiles = str(row['smiles']).strip()
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        img = Draw.MolToImage(mol, size=(300, 300))
        img.save(os.path.join(output_dir, f'smiles_{idx+1}.png'))
    else:
        print(f"Invalid SMILES at row {idx+1}: {smiles}")

print(f"2D images saved to {output_dir}")
