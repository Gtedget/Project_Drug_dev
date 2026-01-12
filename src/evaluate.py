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
