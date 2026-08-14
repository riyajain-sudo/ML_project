#a separte script to inspect the external data before processing it -> E. coli DNA gyrase compounds


import pandas as pd
from tabulate import tabulate

df = pd.read_csv(
    "../data/gyrB_external_raw.csv",
    sep=";",
    quotechar='"',
    engine="c",          # faster + usually more forgiving here than python engine
    on_bad_lines="warn"  # tells you which rows are malformed instead of crashing
)

# Basic information
print("=" * 60)
print("EXTERNAL DATASET INSPECTION")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)


# Show only important columns
important_columns = [
    "Molecule ChEMBL ID",
    "Smiles",
    "Standard Type",
    "Standard Value",
    "Standard Units",
    "pChEMBL Value"
]

print("\nImportant Columns:")
print(important_columns)


# Activity types
print("\nStandard Types:")
print(df["Standard Type"].value_counts(dropna=False))


# Units
print("\nStandard Units:")
print(df["Standard Units"].value_counts(dropna=False))


# Missing values only for important columns
print("\nMissing Values:")
print(df[important_columns].isnull().sum())


# First 10 rows - only important columns
print("\nFirst 10 Records:")
print(
    tabulate(
        df[important_columns].head(10),
        headers="keys",
        tablefmt="grid",
        showindex=False
    )
)