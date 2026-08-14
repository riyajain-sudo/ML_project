#IC50 -> Half Maximal Inhibitory Concentration; the concentration of a compound required to inhibit 50% of a biological target.

import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("../data/gyrB_ml.csv")

print("Minimum IC50 value:", df["standard_value"].min())

# Convert IC50 (nM) to M
df["IC50_M"] = df["standard_value"] * 1e-9

# Calculate pIC50
df["pIC50"] = -np.log10(df["IC50_M"])

# Remove temporary column
df.drop(columns=["IC50_M"], inplace=True)

# Save dataset
df.to_csv("../data/gyrB_pic50.csv", index=False)

print("pIC50 dataset saved successfully!")


# Display first few rows
print(df[["standard_value", "pIC50"]].head())