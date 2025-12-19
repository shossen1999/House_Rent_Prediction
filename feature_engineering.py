# 3_feature_engineering.py
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/cleaned_data.csv")

# Select best features
features = ["Bed", "Bath", "Month", "SubArea", "Region", "PropertyType"]
X = df[features]
y = df["Rent"]

# Categorical columns for CatBoost
categorical_features = ["SubArea", "Region", "PropertyType"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

print("Feature engineering done. Train/test saved.")
