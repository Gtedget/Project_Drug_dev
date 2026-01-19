import pandas as pd
import selfies as sf

df = pd.read_csv("data/filtered_smiles.csv")
df["selfies"] = df["smiles"].apply(sf.encoder)
df.to_csv("data/selfies.csv", index=False)
print(f"Converted {len(df)} SMILES to SELFIES.")
