import pandas as pd
from tabulate import tabulate

df = pd.read_csv("../data/gyrB_raw.csv")

print("Original Shape:", df.shape)

df = df[df["standard_type"] == "IC50"]

print("After keeping IC50:", df.shape)

df = df.dropna(subset=["standard_value"])
print("After removing missing IC50 values:", df.shape)

df = df.dropna(subset=["canonical_smiles"])
print("After removing missing SMILES:", df.shape)

df = df[df["standard_units"] == "nM"]
print("After keeping nM:", df.shape)

df = df.drop_duplicates(subset=["molecule_chembl_id"])
print("After removing duplicates:", df.shape)

df = df[
    [
        "molecule_chembl_id",
        "canonical_smiles",
        "standard_value",
        "standard_units",
        "pchembl_value",
    ]
]

print(df.head())
# print(tabulate(df.head(), headers="keys", tablefmt="grid"))

df.to_csv("../data/gyrB_clean.csv", index=False)

print(df.info())
print(df.describe())

from tabulate import tabulate

print(tabulate(df.head(10), headers="keys", tablefmt="grid"))

