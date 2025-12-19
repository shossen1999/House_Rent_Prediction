# 4_train_model.py
import os
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load data
X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv")
y_test = pd.read_csv("data/y_test.csv")

categorical_features = ["SubArea", "Region", "PropertyType"]

# Ensure directory exists
os.makedirs("models", exist_ok=True)

# Improved CatBoost Model
model = CatBoostRegressor(
    iterations=1200,
    learning_rate=0.04,
    depth=7,
    loss_function="RMSE",
    random_seed=42,
    l2_leaf_reg=3,
    verbose=200
)

model.fit(
    X_train, y_train,
    cat_features=categorical_features,
    eval_set=(X_test, y_test)
)

# Predictions
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("\n====================")
print(f"RMSE: {rmse}")
print(f"MAE:  {mae}")
print("====================\n")

# Save model
MODEL_PATH = "models/catboost_model.cbm"
model.save_model(MODEL_PATH)

print(f"Model saved successfully at: {MODEL_PATH}")
