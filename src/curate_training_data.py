from __future__ import annotations

import argparse
from dataclasses import dataclass

from runtime_bootstrap import bootstrap_runtime

bootstrap_runtime()

import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from sequence_utils import smiles_to_selfies


EXCLUDED_OUTPUT_COLUMNS = {
    "molfile",
    "molregno-2",
    "molregno-3",
    "molregno-4",
    "cpd_str_alert_id",
    "alert_id",
}

PREFERRED_OUTPUT_COLUMNS = [
    "molregno",
    "drug_chembl_id",
    "drug_name",
    "action_type",
    "binding_site",
    "target_chembl_id",
    "target_name",
    "mesh_indication",
    "efo_indication",
    "max_phase_for_ind",
    "mw_freebase",
    "alogp",
    "hba",
    "hbd",
    "psa",
    "rtb",
    "ro3_pass",
    "num_ro5_violations",
    "full_mwt",
    "aromatic_rings",
    "heavy_atoms",
    "qed_weighted",
    "full_molformula",
    "np_likeness_score",
    "standard_inchi",
    "standard_inchi_key",
    "smiles",
    "original_smiles",
    "had_multiple_fragments",
    "was_standardized",
    "smiles_length",
    "target_count_for_smiles",
]


@dataclass(frozen=True)
class CuratedSmiles:
    smiles: str
    had_multiple_fragments: bool
    changed: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a curated training CSV from raw SMILES data.")
    parser.add_argument("--input_csv", type=str, default="data/raw_smiles.csv")
    parser.add_argument("--output_csv", type=str, default="data/curated_training_smiles.csv")
    parser.add_argument("--report_csv", type=str, default="data/curation_report.csv")
    parser.add_argument("--smiles_column", type=str, default="smiles")
    parser.add_argument("--target_column", type=str, default="target_name")
    parser.add_argument("--min_smiles_length", type=int, default=5)
    parser.add_argument("--max_smiles_length", type=int, default=128)
    parser.add_argument("--max_targets_per_smiles", type=int, default=2)
    parser.add_argument("--selfies_column", type=str, default=None)
    return parser.parse_args()


def curate_smiles(
    smiles: str,
    largest_fragment_chooser: rdMolStandardize.LargestFragmentChooser,
    uncharger: rdMolStandardize.Uncharger,
) -> CuratedSmiles | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    curated = rdMolStandardize.Cleanup(mol)
    curated = largest_fragment_chooser.choose(curated)
    curated = uncharger.uncharge(curated)

    curated_smiles = Chem.MolToSmiles(curated, canonical=True)
    original_smiles = Chem.MolToSmiles(mol, canonical=True)
    return CuratedSmiles(
        smiles=curated_smiles,
        had_multiple_fragments="." in smiles,
        changed=curated_smiles != original_smiles,
    )


def build_report_rows(
    total_rows: int,
    non_null_smiles: int,
    invalid_smiles: int,
    standardized_changed: int,
    multi_fragment_rows: int,
    dropped_by_length: int,
    dropped_by_target_ambiguity: int,
    deduped_rows: int,
    output_rows: int,
    unique_smiles: int,
) -> list[dict[str, int]]:
    return [
        {"metric": "input_rows", "value": total_rows},
        {"metric": "rows_with_smiles", "value": non_null_smiles},
        {"metric": "invalid_smiles_removed", "value": invalid_smiles},
        {"metric": "standardized_rows_changed", "value": standardized_changed},
        {"metric": "multifragment_rows_seen", "value": multi_fragment_rows},
        {"metric": "rows_dropped_by_length", "value": dropped_by_length},
        {"metric": "rows_dropped_by_target_ambiguity", "value": dropped_by_target_ambiguity},
        {"metric": "duplicate_rows_removed", "value": deduped_rows},
        {"metric": "output_rows", "value": output_rows},
        {"metric": "output_unique_smiles", "value": unique_smiles},
    ]


def select_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    ordered_columns = [column for column in PREFERRED_OUTPUT_COLUMNS if column in df.columns]
    remaining_columns = [
        column
        for column in df.columns
        if column not in ordered_columns and column not in EXCLUDED_OUTPUT_COLUMNS
    ]
    return df[ordered_columns + remaining_columns].copy()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input_csv)
    total_rows = len(df)
    df = df[df[args.smiles_column].notna()].copy()
    non_null_smiles = len(df)

    chooser = rdMolStandardize.LargestFragmentChooser()
    uncharger = rdMolStandardize.Uncharger()

    curated_rows: list[CuratedSmiles | None] = [
        curate_smiles(str(smiles), chooser, uncharger) for smiles in df[args.smiles_column].astype(str)
    ]

    df["_curated"] = curated_rows
    invalid_smiles = int(df["_curated"].isna().sum())
    df = df[df["_curated"].notna()].copy()

    df["original_smiles"] = df[args.smiles_column].astype(str)
    df["smiles"] = df["_curated"].map(lambda item: item.smiles)
    df["had_multiple_fragments"] = df["_curated"].map(lambda item: item.had_multiple_fragments)
    df["was_standardized"] = df["_curated"].map(lambda item: item.changed)
    standardized_changed = int(df["was_standardized"].sum())
    multi_fragment_rows = int(df["had_multiple_fragments"].sum())
    df = df.drop(columns=["_curated"])

    df["smiles_length"] = df["smiles"].str.len()
    before_length_filter = len(df)
    df = df[
        (df["smiles_length"] >= args.min_smiles_length)
        & (df["smiles_length"] <= args.max_smiles_length)
    ].copy()
    dropped_by_length = before_length_filter - len(df)

    dedupe_columns = ["smiles"]
    if args.target_column in df.columns:
        dedupe_columns.append(args.target_column)
    before_dedupe = len(df)
    df = df.drop_duplicates(subset=dedupe_columns).copy()
    deduped_rows = before_dedupe - len(df)

    dropped_by_target_ambiguity = 0
    if args.target_column in df.columns:
        target_counts = df.groupby("smiles")[args.target_column].nunique().rename("target_count_for_smiles")
        df = df.join(target_counts, on="smiles")
        before_target_filter = len(df)
        df = df[df["target_count_for_smiles"] <= args.max_targets_per_smiles].copy()
        dropped_by_target_ambiguity = before_target_filter - len(df)
    else:
        df["target_count_for_smiles"] = 0

    report_rows = build_report_rows(
        total_rows=total_rows,
        non_null_smiles=non_null_smiles,
        invalid_smiles=invalid_smiles,
        standardized_changed=standardized_changed,
        multi_fragment_rows=multi_fragment_rows,
        dropped_by_length=dropped_by_length,
        dropped_by_target_ambiguity=dropped_by_target_ambiguity,
        deduped_rows=deduped_rows,
        output_rows=len(df),
        unique_smiles=df["smiles"].nunique(),
    )

    if args.selfies_column:
        output_selfies = df["smiles"].map(smiles_to_selfies)
        df[args.selfies_column] = output_selfies
        df = df[df[args.selfies_column].notna()].copy()

    output_df = select_output_columns(df)

    output_df.to_csv(args.output_csv, index=False)
    pd.DataFrame(report_rows).to_csv(args.report_csv, index=False)

    print(f"Saved curated training data to {args.output_csv}")
    print(f"Saved curation report to {args.report_csv}")
    for row in report_rows:
        print(f"{row['metric']}: {row['value']}")


if __name__ == "__main__":
    main()