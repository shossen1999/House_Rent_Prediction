# test.py
# Predict rent using trained CatBoost model and user input

import pandas as pd
from catboost import CatBoostRegressor

MODEL_PATH = "models/catboost_model.cbm"


def get_user_inputs():
    print("\n=== Enter House Details for Rent Prediction ===")

    # Numeric fields
    bed = int(input("Number of Bedrooms (Bed): "))
    bath = int(input("Number of Bathrooms (Bath): "))

    month = int(input("Month (1–12): "))
    if not (1 <= month <= 12):
        print("⚠️ Invalid month! Setting Month = 1")
        month = 1

    # Categorical fields
    subarea = input("SubArea (e.g., Mirpur, Dhanmondi, Gulshan): ").strip().title()
    region = input("Region (e.g., Dhaka, Chittagong): ").strip().title()
    property_type = input("PropertyType (e.g., Family Flat, Apartment): ").strip().title()

    data = {
        "Bed": [bed],
        "Bath": [bath],
        "Month": [month],
        "SubArea": [subarea],
        "Region": [region],
        "PropertyType": [property_type],
    }

    return pd.DataFrame(data)


def main():
    print("\nLoading model...")
    model = CatBoostRegressor()
    model.load_model(MODEL_PATH)

    # Collect data from user
    user_df = get_user_inputs()

    print("\n=== Input Summary ===")
    print(user_df.to_string(index=False))


    print("\nPredicting rent...")
    predicted_rent = model.predict(user_df)[0]

    print("\n==============================")
    print(f"🏠 Predicted Monthly Rent: BDT {predicted_rent:,.0f}")
    print("==============================\n")


if __name__ == "__main__":
    main()
