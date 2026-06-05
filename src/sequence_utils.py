from __future__ import annotations

from pathlib import Path

from rdkit import Chem
from transformers import PreTrainedTokenizerFast

import selfies as sf


SEQUENCE_REPRESENTATIONS = {"smiles", "selfies"}
SELFIES_REGEX = r"\[[^\]]+\]"
SMILES_REGEX = r"\[.*?\]|Br|Cl|Si|Na|Ca|Li|Mg|Al|Sn|Ag|Au|Fe|Zn|[A-Za-z0-9=#\-\+\(\)\\\/ ]"


def canonicalize_smiles(smiles: str) -> str | None:
    normalized = smiles.replace(" ", "")
    mol = Chem.MolFromSmiles(normalized)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def infer_sequence_representation(column_name: str | None, explicit: str | None = None) -> str:
    if explicit is not None and explicit != "auto":
        if explicit not in SEQUENCE_REPRESENTATIONS:
            raise ValueError(f"Unsupported sequence representation '{explicit}'")
        return explicit

    if column_name and "selfies" in str(column_name).lower():
        return "selfies"
    return "smiles"


def get_default_tokenizer_path(representation: str) -> str:
    if representation == "selfies":
        return "tokenizer/selfies_tokenizer.json"
    return "tokenizer/smiles_tokenizer.json"


def load_tokenizer(tokenizer_path: str) -> PreTrainedTokenizerFast:
    return PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_path,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
    )


def smiles_to_selfies(smiles: str) -> str | None:
    canonical = canonicalize_smiles(smiles)
    if canonical is None:
        return None
    try:
        return sf.encoder(canonical)
    except Exception:
        return None


def sequence_to_smiles(sequence: str, representation: str) -> str | None:
    if representation == "selfies":
        try:
            decoded = sf.decoder(sequence.replace(" ", ""))
        except Exception:
            return None
        return canonicalize_smiles(decoded)
    return canonicalize_smiles(sequence)


def ensure_sequence_column(df, source_smiles_column: str, sequence_column: str, representation: str):
    if sequence_column in df.columns:
        return df

    if representation != "selfies":
        raise ValueError(f"Missing required sequence column '{sequence_column}'")
    if source_smiles_column not in df.columns:
        raise ValueError(f"Missing source SMILES column '{source_smiles_column}' needed to derive SELFIES")

    df = df.copy()
    df[sequence_column] = df[source_smiles_column].astype(str).map(smiles_to_selfies)
    return df


def tokenizer_regex_for_representation(representation: str) -> str:
    if representation == "selfies":
        return SELFIES_REGEX
    return SMILES_REGEX


def tokenizer_output_exists(path: str) -> bool:
    return Path(path).exists()