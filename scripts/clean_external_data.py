"""Clean the checked-in semicolon-delimited external export offline."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/gyrB_external_raw.csv"
OUTPUT = ROOT / "data/gyrB_external_clean.csv"
COUNTS = ROOT / "data/gyrB_external_clean_exclusions.json"

def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    names = {"Molecule ChEMBL ID":"molecule_chembl_id", "Smiles":"canonical_smiles",
             "Standard Type":"standard_type", "Standard Relation":"standard_relation",
             "Standard Value":"standard_value", "Standard Units":"standard_units",
             "pChEMBL Value":"pchembl_value", "Molecular Weight":"molecular_weight",
             "Data Validity Comment":"data_validity_comment", "Assay ChEMBL ID":"assay_chembl_id",
             "Assay Description":"assay_description", "Assay Type":"assay_type",
             "Document ChEMBL ID":"document_chembl_id", "Document Year":"document_year",
             "Target ChEMBL ID":"target_chembl_id", "Target Organism":"target_organism",
             "Compound Key":"compound_key"}
    df = df.rename(columns=names).copy()
    required = {"molecule_chembl_id", "canonical_smiles", "standard_type", "standard_relation",
                "standard_value", "standard_units", "data_validity_comment"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"External source is missing required columns: {sorted(missing)}")
    value = pd.to_numeric(df["standard_value"], errors="coerce")
    relation = df["standard_relation"].fillna("").astype(str).str.strip()
    validity = df["data_validity_comment"].fillna("").astype(str).str.strip()
    units = df["standard_units"].fillna("").astype(str).str.strip()
    mw = pd.to_numeric(df.get("molecular_weight", pd.Series(np.nan, index=df.index)), errors="coerce")
    converted = value.where(units.eq("nM"), (value * 1_000_000 / mw).where(units.eq("ug.mL-1") & mw.gt(0)))
    valid_value = np.isfinite(converted) & (converted > 0)
    keep = (df["standard_type"].eq("IC50") & relation.eq("=") & validity.eq("") &
            units.isin(["nM", "ug.mL-1"]) & df["molecule_chembl_id"].notna() &
            df["canonical_smiles"].notna() & valid_value)
    counts = {"input_rows": int(len(df)), "not_IC50": int((~df["standard_type"].eq("IC50")).sum()),
              "not_exact_relation": int((~relation.eq("=")).sum()),
              "invalid_or_flagged": int((~validity.eq("")).sum()),
              "unsupported_units": int((~units.isin(["nM", "ug.mL-1"])).sum()),
              "missing_identity_or_smiles": int((df["molecule_chembl_id"].isna() | df["canonical_smiles"].isna()).sum()),
              "nonpositive_nonfinite_or_unconvertible": int((~valid_value).sum()), "retained_rows": int(keep.sum())}
    wanted = ["activity_id", "molecule_chembl_id", "compound_key", "canonical_smiles",
              "standard_type", "standard_relation", "standard_value", "standard_units",
              "molecular_weight", "IC50_nM", "pchembl_value", "data_validity_comment",
              "assay_chembl_id", "assay_description", "assay_type", "document_chembl_id",
              "document_year", "target_chembl_id", "target_organism"]
    out = df.loc[keep, [c for c in wanted if c in df.columns]].copy()
    out["standard_value"] = value.loc[keep].astype(float)
    out["IC50_nM"] = converted.loc[keep].astype(float)
    out["pIC50"] = -np.log10(out["IC50_nM"] * 1e-9)
    return out, counts

def main() -> None:
    source = pd.read_csv(SOURCE, sep=";", quotechar='"', on_bad_lines="error")
    cleaned, counts = clean(source)
    cleaned.to_csv(OUTPUT, index=False)
    COUNTS.write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(counts, indent=2))
    print(f"Saved external exact-measurement export to {OUTPUT}")

if __name__ == "__main__":
    main()
