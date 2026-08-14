import pandas as pd
import numpy as np

df = pd.read_csv("../data/gyrB_descriptors.csv")

print("Dataset Shape:", df.shape)
print(df.head())

print("\nColumns:")
print(df.columns.tolist())


## Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

ml_df = df.drop(
    columns=[
        "molecule_chembl_id",
        "canonical_smiles",
        "standard_units",
        "pchembl_value",
    ]
)

print("\nML Dataset Shape:", ml_df.shape)

print(ml_df.columns)

print("\nNumber of unique values in each column:")
print(ml_df.nunique())

# automatically identify constant columns
constant_columns = [
    col for col in ml_df.columns
    if ml_df[col].nunique() == 1
]

print("Constant Columns:", constant_columns)

#calculate correlation matrix
correlation_matrix = ml_df.corr()

# print(correlation_matrix)

# Identify highly correlated features
corr_matrix = ml_df.drop(columns=["standard_value"]).corr().abs()

upper = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

high_corr = [
    column
    for column in upper.columns
    if any(upper[column] > 0.90)
]

print("Highly Correlated Columns:")
print(high_corr)

corr_matrix = ml_df.drop(columns=["standard_value"]).corr()

print(corr_matrix["HeavyAtoms"].sort_values(ascending=False))

#removing highly correlated features
ml_df = ml_df.drop(columns=high_corr)

print("\nFinal ML Dataset Shape:", ml_df.shape)
print(ml_df.columns)

#save the final dataset to a new CSV file
ml_df.to_csv("../data/gyrB_ml.csv", index=False)

print("\nFinal ML dataset saved successfully!")