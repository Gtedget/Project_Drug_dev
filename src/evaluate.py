from rdkit import Chem
from rdkit.Chem import QED
import pandas as pd

df = pd.read_csv("data/generated_smiles.csv")

valid = []
for s in df["smiles"]:
    mol = Chem.MolFromSmiles(s)
    if mol:
        valid.append({
            "smiles": s,
            "qed": QED.qed(mol)
        })

pd.DataFrame(valid).to_csv("data/filtered_smiles.csv", index=False)
print(f"Valid molecules: {len(valid)}")

# Also save valid SMILES to generated_smiles_valid.csv
pd.DataFrame(valid)[["smiles"]].to_csv("data/generated_smiles_valid.csv", index=False)
print(f"Valid SMILES saved to data/generated_smiles_valid.csv")
