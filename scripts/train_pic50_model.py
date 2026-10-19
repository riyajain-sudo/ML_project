from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import pandas as pd
from sklearn.model_selection import train_test_split   #evaluates every parameter combination using cross-validation.
from sklearn.ensemble import (
    RandomForestRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

import matplotlib.pyplot as plt

import joblib
         

#Load dataset
df = pd.read_csv(ROOT / "data/gyrB_pic50.csv")

X = df.drop(columns=["standard_value", "pIC50"])
y = df["pIC50"]

# print("Features Shape:", X.shape)
# print("Target Shape:", y.shape)

# print("\nFeatures:")
# print(X.head())

# print("\nTarget:")
# print(y.head())

#train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# print("Training Features:", X_train.shape)
# print("Testing Features:", X_test.shape)

# print("Training Target:", y_train.shape)
# print("Testing Target:", y_test.shape)

#creating the model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42              #data doesnt change
)


#training the model
model.fit(X_train, y_train)

#saving the model
joblib.dump(model, ROOT / "models/gyrB_rf_pic50.pkl")

print("Model saved successfully!")

#prediction
y_pred = model.predict(X_test)

# print("Predicted IC50 values:")
# print(y_pred[:10])

# print("\nActual IC50 values:")
# print(y_test.iloc[:10].values)


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.3f}")


#check score of the model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print("Training Score:", train_score)
print("Testing Score:", test_score)


#Feature importance -> which feature is playing the most important role in prediction
# importance = model.feature_importances_

# feature_importance = pd.DataFrame({
#     "Feature": X.columns,
#     "Importance": importance
# })

# feature_importance = feature_importance.sort_values(
#     by="Importance",
#     ascending=False
# )

# print("\nFeature Importance:")
# print(feature_importance)

# #plotting graph
# plt.figure(figsize=(8,5))

# plt.bar(
#     feature_importance["Feature"],
#     feature_importance["Importance"]
# )

# plt.xlabel("Descriptors")
# plt.ylabel("Importance")
# plt.title("Random Forest Feature Importance")

# plt.xticks(rotation=45)

# plt.tight_layout()

# plt.show()


# Scatter Plot
plt.figure(figsize=(6,6))

plt.scatter(y_test, y_pred, alpha=0.7)

# Ideal prediction line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--',
    linewidth=2
)

plt.xlabel("Actual pIC50")
plt.ylabel("Predicted pIC50")
plt.title("Actual vs Predicted pIC50")

plt.tight_layout()
# plt.show()


# Residual Plot

residuals = y_test - y_pred

plt.figure(figsize=(7,5))

plt.scatter(
    y_pred,
    residuals,
    alpha=0.7,
    edgecolors="black"
)

plt.axhline(
    y=0,
    color="red",
    linestyle="--"
)

plt.xlabel("Predicted pIC50")
plt.ylabel("Residuals")
plt.title("Residual Plot")

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig("residual_plot.png", dpi=300)

# plt.show()