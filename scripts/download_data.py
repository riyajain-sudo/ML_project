from chembl_webresource_client.new_client import new_client
import pandas as pd

activity = new_client.activity

result = activity.filter(
    target_chembl_id="CHEMBL1921666"
)

df = pd.DataFrame(result)

# print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nActivity Types:")
print(df["standard_type"].value_counts())

print("\nUnits:")
print(df["standard_units"].value_counts())

print("\nMissing Values:")
print(df.isnull().sum())

# Save the raw dataset
df.to_csv("../data/gyrB_raw.csv", index=False)

print("Raw dataset saved successfully!")