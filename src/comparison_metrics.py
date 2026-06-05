from __future__ import annotations

from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, QED

from conditional_hybrid_graph_generator import DEFAULT_DESCRIPTOR_NAMES, compute_descriptor_value


def load_train_smiles(train_csv: str, smiles_column: str = "smiles") -> set[str]:
    df = pd.read_csv(train_csv)
    if smiles_column not in df.columns:
        return set()
    return {
        Chem.MolToSmiles(Chem.MolFromSmiles(str(smiles)))
        for smiles in df[smiles_column].dropna()
        if Chem.MolFromSmiles(str(smiles)) is not None
    }


def evaluate_generated_file(
    generated_csv: str,
    requested_samples: int,
    train_smiles: set[str],
    smiles_column: str = "smiles",
) -> dict[str, float | int | str]:
    empty_metrics = {
        "output_csv": generated_csv,
        "requested_samples": requested_samples,
        "valid_count": 0,
        "validity_rate": 0.0,
        "unique_count": 0,
        "uniqueness_rate": 0.0,
        "novel_count": 0,
        "novelty_rate": 0.0,
        "mean_qed": 0.0,
        "mean_logp": 0.0,
        "mean_mol_wt": 0.0,
        "logp_target_mae": 0.0,
        "descriptor_target_mae": 0.0,
        "descriptor_target_coverage": 0.0,
    }

    path = Path(generated_csv)
    if not path.exists():
        return empty_metrics

    df = pd.read_csv(path)
    if smiles_column not in df.columns or df.empty:
        return empty_metrics

    canonical_smiles: list[str] = []
    qeds: list[float] = []
    logps: list[float] = []
    mol_wts: list[float] = []
    logp_target_errors: list[float] = []
    descriptor_target_errors: list[float] = []
    descriptor_rows_scored = 0

    target_logp_column = "condition_alogp" if "condition_alogp" in df.columns else ("alogp" if "alogp" in df.columns else None)
    descriptor_target_columns = {
        descriptor_name: f"condition_{descriptor_name}"
        for descriptor_name in DEFAULT_DESCRIPTOR_NAMES
        if f"condition_{descriptor_name}" in df.columns
    }

    for _, row in df.iterrows():
        smiles = row.get(smiles_column)
        if pd.isna(smiles):
            continue

        smiles_str = str(smiles)
        mol = Chem.MolFromSmiles(smiles_str)
        if mol is None:
            continue
        canonical = Chem.MolToSmiles(mol)
        canonical_smiles.append(canonical)
        qeds.append(QED.qed(mol))
        current_logp = Crippen.MolLogP(mol)
        logps.append(current_logp)
        mol_wts.append(Descriptors.MolWt(mol))

        if target_logp_column is not None and not pd.isna(row.get(target_logp_column)):
            logp_target_errors.append(abs(current_logp - float(row[target_logp_column])))

        row_descriptor_errors: list[float] = []
        for descriptor_name, column_name in descriptor_target_columns.items():
            if pd.isna(row.get(column_name)):
                continue
            generated_value = compute_descriptor_value(descriptor_name, smiles=smiles_str)
            row_descriptor_errors.append(abs(generated_value - float(row[column_name])))

        if row_descriptor_errors:
            descriptor_rows_scored += 1
            descriptor_target_errors.extend(row_descriptor_errors)

    valid_count = len(canonical_smiles)
    unique_smiles = set(canonical_smiles)
    unique_count = len(unique_smiles)
    novel_count = sum(smiles not in train_smiles for smiles in unique_smiles)

    return {
        "output_csv": generated_csv,
        "requested_samples": requested_samples,
        "valid_count": valid_count,
        "validity_rate": valid_count / max(requested_samples, 1),
        "unique_count": unique_count,
        "uniqueness_rate": unique_count / max(valid_count, 1),
        "novel_count": novel_count,
        "novelty_rate": novel_count / max(unique_count, 1),
        "mean_qed": sum(qeds) / max(len(qeds), 1),
        "mean_logp": sum(logps) / max(len(logps), 1),
        "mean_mol_wt": sum(mol_wts) / max(len(mol_wts), 1),
        "logp_target_mae": sum(logp_target_errors) / max(len(logp_target_errors), 1),
        "descriptor_target_mae": sum(descriptor_target_errors) / max(len(descriptor_target_errors), 1),
        "descriptor_target_coverage": descriptor_rows_scored / max(valid_count, 1),
    }