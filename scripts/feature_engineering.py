"""Build a numeric-only model matrix and a row-aligned provenance companion."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/gyrB_descriptors.csv"
MODEL_OUTPUT = ROOT / "data/gyrB_ml.csv"
PROVENANCE_OUTPUT = ROOT / "data/gyrB_ml_provenance.csv"
DESCRIPTORS = ["MolecularWeight", "LogP", "TPSA", "HBD", "HBA",
               "RotatableBonds", "RingCount", "HeavyAtoms"]
PROVENANCE = ["activity_id", "record_id", "molecule_chembl_id", "parent_molecule_chembl_id",
              "canonical_smiles", "standard_type", "standard_relation", "standard_units",
              "data_validity_comment", "data_validity_description", "assay_chembl_id",
              "assay_description", "assay_type", "document_chembl_id", "document_year",
              "target_chembl_id"]

def build_tables(df):
    required = set(DESCRIPTORS + ["standard_value"])
    missing = required - set(df.columns)
    if missing: raise ValueError(f"Descriptor input missing columns: {sorted(missing)}")
    model = df[DESCRIPTORS + ["standard_value"]].apply(pd.to_numeric, errors="coerce")
    if model.isna().any().any():
        raise ValueError("Descriptor/target matrix contains missing or non-numeric values")
    provenance = df[[c for c in PROVENANCE if c in df.columns]].copy()
    provenance.insert(0, "source_row", range(len(df)))
    return model, provenance

def main():
    frame = pd.read_csv(SOURCE)
    model, provenance = build_tables(frame)
    model.to_csv(MODEL_OUTPUT, index=False)
    provenance.to_csv(PROVENANCE_OUTPUT, index=False)
    print(f"Saved {len(model)} numeric model rows to {MODEL_OUTPUT}; row-aligned identity/assay provenance to {PROVENANCE_OUTPUT}")

if __name__ == "__main__": main()
