"""Convert positive finite exact nM IC50 values to pIC50."""
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/gyrB_ml.csv"
OUTPUT = ROOT / "data/gyrB_pic50.csv"

def convert(df: pd.DataFrame) -> pd.DataFrame:
    required = {"standard_value", "standard_units"}
    if not required <= set(df.columns):
        raise ValueError(f"Missing required columns: {sorted(required - set(df.columns))}")
    values = pd.to_numeric(df["standard_value"], errors="coerce")
    valid = np.isfinite(values) & values.gt(0) & df["standard_units"].eq("nM")
    if not valid.all():
        raise ValueError(f"Refusing pIC50 conversion: {int((~valid).sum())} non-positive, non-finite, or non-nM row(s)")
    out = df.copy()
    out["pIC50"] = -np.log10(values * 1e-9)
    return out

def main():
    frame = pd.read_csv(SOURCE)
    provenance_path = ROOT / "data/gyrB_ml_provenance.csv"
    provenance = pd.read_csv(provenance_path)
    if len(frame) != len(provenance) or not provenance["source_row"].reset_index(drop=True).equals(pd.Series(range(len(frame)), name="source_row")):
        raise ValueError("Model rows and provenance sidecar do not align; regenerate feature tables")
    if not provenance["standard_relation"].eq("=").all() or not provenance["standard_units"].eq("nM").all():
        raise ValueError("pIC50 requires exact nM measurements in the aligned provenance sidecar")
    if provenance["data_validity_comment"].fillna("").astype(str).str.strip().ne("").any():
        raise ValueError("pIC50 refuses source rows with data-validity flags")
    frame["standard_units"] = provenance["standard_units"].values
    frame = convert(frame).drop(columns=["standard_units"])
    frame.to_csv(OUTPUT, index=False)
    print(f"Converted {len(frame)} exact positive nM IC50 values to {OUTPUT}")

if __name__ == "__main__": main()
