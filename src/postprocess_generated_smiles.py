from rdkit import Chem
from rdkit.Chem import QED, Descriptors
import pandas as pd

df = pd.read_csv("data/generated_smiles.csv")
valid = []
for s in df["smiles"]:
    mol = Chem.MolFromSmiles(s)
    if mol:
        mw = Descriptors.MolWt(mol)
        qed = QED.qed(mol)
        # Example drug-likeness rules
        if 200 < mw < 500 and qed > 0.5:
            valid.append({"smiles": s, "qed": qed, "molwt": mw})

df_valid = pd.DataFrame(valid).drop_duplicates(subset="smiles")
df_valid.to_csv("data/generated_smiles_filtered.csv", index=False)
print(f"Filtered valid, drug-like SMILES: {len(df_valid)}")
