# 2_preprocessing.py
import pandas as pd

FILE_PATH = "data/rent_data.csv"
df = pd.read_csv(FILE_PATH)

# -------------------------
# Clean Bed & Bath
# -------------------------
df["Bed"] = pd.to_numeric(df["Bed"], errors="coerce").fillna(0).astype(int)
df["Bath"] = pd.to_numeric(df["Bath"], errors="coerce").fillna(1).astype(int)

# -------------------------
# Extract Month from "To-let From"
# -------------------------
df["Month"] = df["To-let From"].str.extract(r"(\w+)").fillna("")
month_map = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10,
    "November": 11, "December": 12
}
df["Month"] = df["Month"].map(month_map).fillna(0).astype(int)

# -------------------------
# Extract SubArea from Location
# Example: "Mirpur, Dhaka" → Mirpur
# -------------------------
df["SubArea"] = df["Location"].astype(str).str.split(",").str[0].str.strip()

# -------------------------
# Region column is already present
# Just clean it
# -------------------------
df["Region"] = df["Region"].astype(str).str.strip()

# -------------------------
# Clean property type
# -------------------------
def clean_property_type(x):
    x = str(x).lower()
    if "family" in x:
        return "Family"
    if "bachelor" in x and "room" in x:
        return "Bachelor Room"
    if "seat" in x:
        return "Bachelor Seat"
    if "sublet" in x:
        return "Sublet"
    if "room" in x:
        return "Room"
    if "flat" in x or "apartment" in x:
        return "Flat"
    return "Other"

df["PropertyType"] = df["Title"].apply(clean_property_type)

# -------------------------
# Remove extreme outliers
# -------------------------
df = df[df["Rent"] < 150000]  # rents above 150k distort model

# -------------------------
# Save cleaned dataset
# -------------------------
df.to_csv("data/cleaned_data.csv", index=False)
print("Preprocessing completed. Saved cleaned_data.csv")
