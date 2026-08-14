#do not give standard value as input to the model, only give pIC50 values as input to the model to avoid DATA LEAKAGE

import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor


# Load pIC50 dataset
df = pd.read_csv("../data/gyrB_pic50.csv")


# Features
features = [
    "MolecularWeight",
    "LogP",
    "TPSA",
    "HBD",
    "HBA",
    "RotatableBonds",
    "RingCount"
]

X = df[features]

# Target
y = df["pIC50"]


# Create final Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


# Train on ALL 333 compounds
model.fit(X, y)


# Save the final model
joblib.dump(
    model,
    "../models/gyrB_rf_pic50_final.pkl"
)


print("Final model trained successfully!")
print("Training compounds:", len(df))
print("Features used:", features)
print("Model saved successfully!")