"""Prepare exact, valid nM IC50 measurements while retaining provenance."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/gyrB_raw.csv"
OUTPUT = ROOT / "data/gyrB_clean.csv"
COUNTS = ROOT / "data/gyrB_clean_exclusions.json"

def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    required = {"standard_type", "standard_relation", "standard_value", "standard_units",
                "canonical_smiles", "molecule_chembl_id", "data_validity_comment"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Source is missing required columns: {sorted(missing)}")
    value = pd.to_numeric(df["standard_value"], errors="coerce")
    relation = df["standard_relation"].fillna("").astype(str).str.strip()
    validity = df["data_validity_comment"].fillna("").astype(str).str.strip()
    valid_value = np.isfinite(value) & (value > 0)
    keep = (df["standard_type"].eq("IC50") & relation.eq("=") & validity.eq("") &
            df["standard_units"].eq("nM") & df["canonical_smiles"].notna() &
            df["molecule_chembl_id"].notna() & valid_value)
    counts = {
        "input_rows": int(len(df)),
        "not_IC50": int((~df["standard_type"].eq("IC50")).sum()),
        "not_exact_relation": int((~relation.eq("=")).sum()),
        "invalid_or_flagged": int((~validity.eq("")).sum()),
        "not_nM": int((~df["standard_units"].eq("nM")).sum()),
        "missing_identity_or_smiles": int((df["canonical_smiles"].isna() | df["molecule_chembl_id"].isna()).sum()),
        "nonpositive_or_nonfinite_value": int((~valid_value).sum()),
        "retained_rows": int(keep.sum()),
    }
    # Keep source identifiers and assay/document/value metadata for traceability.
    wanted = ["activity_id", "record_id", "molecule_chembl_id", "parent_molecule_chembl_id",
              "canonical_smiles", "standard_type", "standard_relation", "standard_value",
              "standard_units", "standard_upper_value", "pchembl_value", "data_validity_comment",
              "data_validity_description", "assay_chembl_id", "assay_description", "assay_type",
              "document_chembl_id", "document_year", "target_chembl_id"]
    kept = df.loc[keep, [c for c in wanted if c in df.columns]].copy()
    kept["standard_value"] = value.loc[keep].astype(float)
    return kept, counts

def main() -> None:
    source = pd.read_csv(SOURCE)
    cleaned, counts = clean(source)
    cleaned.to_csv(OUTPUT, index=False)
    COUNTS.write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(counts, indent=2))
    print(f"Saved provenance-preserving exact IC50 measurements to {OUTPUT}")

if __name__ == "__main__":
    main()
