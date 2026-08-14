#didnt work; 

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV   #evaluates every parameter combination using cross-validation.
from sklearn.ensemble import (
    RandomForestRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

#Load dataset
df = pd.read_csv("../data/gyrB_ml.csv")

#divide the dataset into features and target variable
X = df.drop(columns=["standard_value"])
y = df["standard_value"]

#train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create base Random Forest
model = RandomForestRegressor(random_state=42)

#GridSearchCV -> find the best performing tree; Try every possible combination of these values; 5-Fold Cross Validation is used
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

# Train
grid_search.fit(X_train, y_train)

# Print best parameters
print(grid_search.best_params_)

# Print best CV score
print(grid_search.best_score_)

# Get best model
best_model = grid_search.best_estimator_

# Evaluate on train/test sets
print(best_model.score(X_train, y_train))
print(best_model.score(X_test, y_test))