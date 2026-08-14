import requests
import pandas as pd


target_id = "CHEMBL2094139"

url = (
    "https://www.ebi.ac.uk/chembl/api/data/activity.json"
    f"?target_chembl_id={target_id}"
    "&limit=1000"
)

response = requests.get(url)

print("Status Code:", response.status_code)

if response.status_code != 200:
    print("Failed to retrieve data.")
    print(response.text[:500])
    exit()

data = response.json()

df = pd.DataFrame(data["activities"])

print("\nDataset Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nActivity Types:")
print(df["standard_type"].value_counts())

print("\nUnits:")
print(df["standard_units"].value_counts())

print("\nFirst 5 rows:")
print(df.head())