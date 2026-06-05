import argparse

from runtime_bootstrap import bootstrap_runtime

bootstrap_runtime()

import pandas as pd

from sequence_utils import smiles_to_selfies


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Add a SELFIES column derived from a SMILES column.")
	parser.add_argument("--input_csv", type=str, default="data/filtered_smiles.csv")
	parser.add_argument("--output_csv", type=str, default="data/selfies.csv")
	parser.add_argument("--smiles_column", type=str, default="smiles")
	parser.add_argument("--selfies_column", type=str, default="selfies")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	df = pd.read_csv(args.input_csv)
	if args.smiles_column not in df.columns:
		raise ValueError(f"Missing required SMILES column '{args.smiles_column}'")

	df[args.selfies_column] = df[args.smiles_column].astype(str).map(smiles_to_selfies)
	df = df[df[args.selfies_column].notna()].copy()
	df.to_csv(args.output_csv, index=False)
	print(f"Converted {len(df)} rows to SELFIES in column '{args.selfies_column}'.")


if __name__ == "__main__":
	main()
