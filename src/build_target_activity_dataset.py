from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from typing import Any

from runtime_bootstrap import bootstrap_runtime

bootstrap_runtime()

import pandas as pd

from sequence_utils import smiles_to_selfies


DEFAULT_STANDARD_TYPES = ("IC50", "Ki", "Kd", "EC50", "Potency")
DEFAULT_STANDARD_RELATIONS = ("=", "<", "<=")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a target-aware training dataset by pulling ChEMBL activity records for the targets already "
            "present in an input CSV and enriching them with UniProt sequence and site annotations."
        )
    )
    parser.add_argument("--input_csv", type=str, default="data/raw_smiles.csv")
    parser.add_argument("--output_csv", type=str, default="data/target_activity_training.csv")
    parser.add_argument("--target_metadata_csv", type=str, default="data/target_metadata_enriched.csv")
    parser.add_argument("--target_column", type=str, default="target_chembl_id")
    parser.add_argument("--target_name_column", type=str, default="target_name")
    parser.add_argument("--max_targets", type=int, default=None)
    parser.add_argument("--max_activities_per_target", type=int, default=100)
    parser.add_argument("--page_size", type=int, default=1000)
    parser.add_argument("--min_confidence_score", type=int, default=5)
    parser.add_argument("--standard_types", type=str, default=",".join(DEFAULT_STANDARD_TYPES))
    parser.add_argument("--allowed_relations", type=str, default=",".join(DEFAULT_STANDARD_RELATIONS))
    parser.add_argument("--selfies_column", type=str, default="selfies")
    parser.add_argument("--save_every_n_targets", type=int, default=10)
    parser.add_argument("--failed_targets_csv", type=str, default="data/target_activity_failed_targets.csv")
    return parser.parse_args()


def fetch_json(url: str, retries: int = 3, backoff_seconds: float = 1.0) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # pragma: no cover - network robustness path
            last_error = exc
            if attempt == retries - 1:
                raise
            time.sleep(backoff_seconds * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def build_chembl_url(endpoint: str, **params: Any) -> str:
    query = urllib.parse.urlencode({key: value for key, value in params.items() if value is not None})
    return f"https://www.ebi.ac.uk/chembl/api/data/{endpoint}.json?{query}"


def build_uniprot_entry_url(accession: str) -> str:
    return f"https://rest.uniprot.org/uniprotkb/{urllib.parse.quote(accession)}.json"


def extract_primary_accession(target_record: dict[str, Any]) -> str | None:
    for component in target_record.get("target_components", []):
        accession = component.get("accession")
        if accession:
            return str(accession)
    return None


def extract_gene_names(uniprot_record: dict[str, Any]) -> str:
    genes = []
    for gene in uniprot_record.get("genes", []):
        gene_name = gene.get("geneName", {}).get("value")
        if gene_name:
            genes.append(str(gene_name))
    return ";".join(sorted(set(genes)))


def extract_pdb_ids(uniprot_record: dict[str, Any]) -> str:
    pdb_ids = []
    for ref in uniprot_record.get("uniProtKBCrossReferences", []):
        if ref.get("database") == "PDB" and ref.get("id"):
            pdb_ids.append(str(ref["id"]))
    return ";".join(sorted(set(pdb_ids)))


def extract_feature_counts(uniprot_record: dict[str, Any]) -> dict[str, int]:
    counts = {
        "binding_site_count": 0,
        "active_site_count": 0,
        "site_feature_count": 0,
    }
    for feature in uniprot_record.get("features", []):
        feature_type = str(feature.get("type", "")).lower()
        if feature_type == "binding site":
            counts["binding_site_count"] += 1
        if feature_type == "active site":
            counts["active_site_count"] += 1
        if feature_type in {"binding site", "active site", "site"}:
            counts["site_feature_count"] += 1
    return counts


def extract_protein_name(uniprot_record: dict[str, Any]) -> str | None:
    recommended = uniprot_record.get("proteinDescription", {}).get("recommendedName", {})
    full_name = recommended.get("fullName", {}).get("value")
    if full_name:
        return str(full_name)
    submission_names = uniprot_record.get("proteinDescription", {}).get("submissionNames", [])
    for submission_name in submission_names:
        full_name = submission_name.get("fullName", {}).get("value")
        if full_name:
            return str(full_name)
    return None


def fetch_target_metadata(target_chembl_id: str, fallback_target_name: str | None = None) -> dict[str, Any]:
    target_url = f"https://www.ebi.ac.uk/chembl/api/data/target/{urllib.parse.quote(target_chembl_id)}.json"
    target_record = fetch_json(target_url)
    accession = extract_primary_accession(target_record)

    uniprot_record: dict[str, Any] = {}
    if accession:
        try:
            uniprot_record = fetch_json(build_uniprot_entry_url(accession))
        except Exception:
            uniprot_record = {}

    feature_counts = extract_feature_counts(uniprot_record) if uniprot_record else {
        "binding_site_count": 0,
        "active_site_count": 0,
        "site_feature_count": 0,
    }

    sequence = uniprot_record.get("sequence", {}).get("value") if uniprot_record else None
    sequence_length = uniprot_record.get("sequence", {}).get("length") if uniprot_record else None

    return {
        "target_chembl_id": target_chembl_id,
        "target_name": fallback_target_name or target_record.get("pref_name"),
        "target_pref_name": target_record.get("pref_name"),
        "target_type": target_record.get("target_type"),
        "target_organism": target_record.get("organism"),
        "uniprot_accession": accession,
        "uniprot_id": uniprot_record.get("uniProtkbId") if uniprot_record else None,
        "protein_name": extract_protein_name(uniprot_record) if uniprot_record else None,
        "gene_names": extract_gene_names(uniprot_record) if uniprot_record else "",
        "protein_sequence": sequence,
        "sequence_length": sequence_length,
        "binding_site_count": feature_counts["binding_site_count"],
        "active_site_count": feature_counts["active_site_count"],
        "site_feature_count": feature_counts["site_feature_count"],
        "pdb_ids": extract_pdb_ids(uniprot_record) if uniprot_record else "",
    }


def iter_target_activities(
    target_chembl_id: str,
    standard_types: set[str],
    allowed_relations: set[str],
    min_confidence_score: int,
    max_activities_per_target: int,
    page_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0

    while len(rows) < max_activities_per_target:
        url = build_chembl_url(
            "activity",
            target_chembl_id=target_chembl_id,
            limit=page_size,
            offset=offset,
        )
        payload = fetch_json(url)
        activities = payload.get("activities", [])
        if not activities:
            break

        for record in activities:
            standard_type = str(record.get("standard_type") or "")
            relation = str(record.get("standard_relation") or "=")
            confidence_score = record.get("confidence_score")
            smiles = record.get("canonical_smiles")
            standard_value = record.get("standard_value")

            if standard_type not in standard_types:
                continue
            if relation not in allowed_relations:
                continue
            if confidence_score not in (None, ""):
                try:
                    if int(confidence_score) < min_confidence_score:
                        continue
                except (TypeError, ValueError):
                    pass
            if not smiles or standard_value in (None, ""):
                continue

            rows.append({
                "target_chembl_id": target_chembl_id,
                "assay_chembl_id": record.get("assay_chembl_id"),
                "assay_type": record.get("assay_type"),
                "assay_description": record.get("assay_description"),
                "assay_organism": record.get("assay_organism"),
                "bao_endpoint": record.get("bao_endpoint"),
                "bao_format": record.get("bao_format"),
                "document_chembl_id": record.get("document_chembl_id"),
                "molecule_chembl_id": record.get("molecule_chembl_id"),
                "smiles": record.get("canonical_smiles"),
                "standard_type": standard_type,
                "standard_relation": relation,
                "standard_value": record.get("standard_value"),
                "standard_units": record.get("standard_units"),
                "pchembl_value": record.get("pchembl_value"),
                "confidence_score": record.get("confidence_score"),
                "activity_comment": record.get("activity_comment"),
                "action_type": record.get("action_type"),
                "data_validity_comment": record.get("data_validity_comment"),
            })
            if len(rows) >= max_activities_per_target:
                break

        page_meta = payload.get("page_meta", {})
        returned = len(activities)
        offset += returned
        if returned == 0 or offset >= int(page_meta.get("total_count", 0)):
            break

    return rows


def main() -> None:
    args = parse_args()

    input_df = pd.read_csv(args.input_csv)
    if args.target_column not in input_df.columns:
        raise ValueError(f"Missing target identifier column '{args.target_column}' in {args.input_csv}")

    target_lookup = (
        input_df[[args.target_column, args.target_name_column]]
        .dropna(subset=[args.target_column])
        .drop_duplicates(subset=[args.target_column])
        .set_index(args.target_column)[args.target_name_column]
        .to_dict()
        if args.target_name_column in input_df.columns
        else {}
    )

    target_ids = [str(value) for value in input_df[args.target_column].dropna().astype(str).drop_duplicates().tolist()]
    if args.max_targets is not None:
        target_ids = target_ids[: args.max_targets]

    standard_types = {item.strip() for item in args.standard_types.split(",") if item.strip()}
    allowed_relations = {item.strip() for item in args.allowed_relations.split(",") if item.strip()}

    target_metadata_rows: list[dict[str, Any]] = []
    activity_rows: list[dict[str, Any]] = []
    failed_targets: list[dict[str, Any]] = []

    def save_outputs() -> None:
        target_metadata_df = pd.DataFrame(target_metadata_rows)
        if not target_metadata_df.empty:
            target_metadata_df = target_metadata_df.drop_duplicates(subset=["target_chembl_id"])
            target_metadata_df.to_csv(args.target_metadata_csv, index=False)

        failed_targets_df = pd.DataFrame(failed_targets)
        if not failed_targets_df.empty:
            failed_targets_df.to_csv(args.failed_targets_csv, index=False)

        activity_df = pd.DataFrame(activity_rows)
        if activity_df.empty:
            return

        activity_df = activity_df.drop_duplicates(
            subset=[
                "target_chembl_id",
                "molecule_chembl_id",
                "assay_chembl_id",
                "standard_type",
                "standard_relation",
                "standard_value",
            ]
        ).copy()

        activity_df["smiles_length"] = activity_df["smiles"].astype(str).str.len()
        if args.selfies_column:
            activity_df[args.selfies_column] = activity_df["smiles"].map(smiles_to_selfies)
            activity_df = activity_df[activity_df[args.selfies_column].notna()].copy()

        ordered_columns = [
            "target_chembl_id",
            "target_name",
            "target_pref_name",
            "target_type",
            "target_organism",
            "uniprot_accession",
            "uniprot_id",
            "protein_name",
            "gene_names",
            "sequence_length",
            "binding_site_count",
            "active_site_count",
            "site_feature_count",
            "pdb_ids",
            "molecule_chembl_id",
            "smiles",
            args.selfies_column if args.selfies_column else None,
            "assay_chembl_id",
            "assay_type",
            "assay_description",
            "assay_organism",
            "bao_endpoint",
            "bao_format",
            "document_chembl_id",
            "standard_type",
            "standard_relation",
            "standard_value",
            "standard_units",
            "pchembl_value",
            "confidence_score",
            "activity_comment",
            "action_type",
            "data_validity_comment",
            "smiles_length",
            "protein_sequence",
        ]
        ordered_columns = [column for column in ordered_columns if column is not None and column in activity_df.columns]
        remaining_columns = [column for column in activity_df.columns if column not in ordered_columns]
        activity_df = activity_df[ordered_columns + remaining_columns].copy()
        activity_df.to_csv(args.output_csv, index=False)

    for index, target_chembl_id in enumerate(target_ids, start=1):
        fallback_target_name = target_lookup.get(target_chembl_id)
        try:
            target_metadata = fetch_target_metadata(target_chembl_id, fallback_target_name=fallback_target_name)
            target_metadata_rows.append(target_metadata)

            rows = iter_target_activities(
                target_chembl_id=target_chembl_id,
                standard_types=standard_types,
                allowed_relations=allowed_relations,
                min_confidence_score=args.min_confidence_score,
                max_activities_per_target=args.max_activities_per_target,
                page_size=args.page_size,
            )
            for row in rows:
                row.update(target_metadata)
            activity_rows.extend(rows)

            print(
                f"[{index}/{len(target_ids)}] {target_chembl_id}: "
                f"{len(rows)} activities, accession={target_metadata.get('uniprot_accession') or 'n/a'}"
            , flush=True)
        except Exception as exc:  # pragma: no cover - network robustness path
            failed_targets.append({
                "target_chembl_id": target_chembl_id,
                "target_name": fallback_target_name,
                "error": repr(exc),
            })
            print(
                f"[{index}/{len(target_ids)}] {target_chembl_id}: failed with {exc}",
                flush=True,
            )

        if args.save_every_n_targets > 0 and index % args.save_every_n_targets == 0:
            save_outputs()
        time.sleep(0.05)

    save_outputs()

    target_metadata_df = pd.DataFrame(target_metadata_rows).drop_duplicates(subset=["target_chembl_id"])
    activity_df = pd.read_csv(args.output_csv) if pd.io.common.file_exists(args.output_csv) else pd.DataFrame()

    if activity_df.empty:
        raise RuntimeError("No activity rows were collected. Try lowering the confidence filter or widening allowed standard types.")

    print(f"Saved target metadata to {args.target_metadata_csv}")
    print(f"Saved enriched activity training data to {args.output_csv}")
    if failed_targets:
        print(f"Saved failed target log to {args.failed_targets_csv}")
        print(f"Failed targets: {len(failed_targets)}")
    print(f"Targets processed: {len(target_metadata_df)}")
    print(f"Activity rows: {len(activity_df)}")
    print(f"Unique molecules: {activity_df['molecule_chembl_id'].nunique() if 'molecule_chembl_id' in activity_df.columns else 0}")


if __name__ == "__main__":
    main()