import pandas as pd
import numpy as np


# Load external dataset
df = pd.read_csv(
    "../data/gyrB_external_raw.csv",
    sep=";",
    quotechar='"',
    engine="c",
    on_bad_lines="warn"
)

print("Original Shape:", df.shape)


# Rename important columns
df = df.rename(columns={
    "Molecule ChEMBL ID": "molecule_chembl_id",
    "Smiles": "canonical_smiles",
    "Standard Type": "standard_type",
    "Standard Value": "standard_value",
    "Standard Units": "standard_units",
    "pChEMBL Value": "pchembl_value",
    "Molecular Weight": "molecular_weight"
})


# Remove missing IC50 values
df = df.dropna(subset=["standard_value"])

print("After removing missing IC50:", df.shape)


# Remove missing units
df = df.dropna(subset=["standard_units"])

print("After removing missing units:", df.shape)


# Keep only the two units we know how to handle
df = df[
    df["standard_units"].isin(["nM", "ug.mL-1"])
]

print("After keeping usable units:", df.shape)


# Convert IC50 to nM
def convert_to_nm(row):

    value = row["standard_value"]
    unit = row["standard_units"]

    if unit == "nM":
        return value

    elif unit == "ug.mL-1":

        mw = row["molecular_weight"]

        if pd.isna(mw) or mw <= 0:
            return np.nan

        return (value * 1_000_000) / mw

    return np.nan


df["IC50_nM"] = df.apply(convert_to_nm, axis=1)


# Remove records where conversion failed
df = df.dropna(subset=["IC50_nM"])


# Calculate pIC50
df["pIC50"] = -np.log10(df["IC50_nM"] * 1e-9)


print("\nFirst 10 converted records:")

print(
    df[
        [
            "molecule_chembl_id",
            "standard_value",
            "standard_units",
            "molecular_weight",
            "IC50_nM",
            "pIC50"
        ]
    ].head(10)
)


# Save
df.to_csv(
    "../data/gyrB_external_clean.csv",
    index=False
)

print("\nClean external dataset saved successfully!")
print("Final Shape:", df.shape)