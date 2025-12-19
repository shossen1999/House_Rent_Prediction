# 1_eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

FILE_PATH = "data/rent_data.csv"

df = pd.read_csv(FILE_PATH)

print("\n===== FIRST 10 ROWS =====")
print(df.head(10))

print("\n===== BASIC INFO =====")
print(df.info())

print("\n===== NULL VALUE COUNT =====")
print(df.isnull().sum())

print("\n===== RENT STATS =====")
print(df["Rent"].describe())

# Remove extreme outliers for visualization
df_plot = df[df["Rent"] < 150000]

plt.figure(figsize=(8,5))
sns.histplot(df_plot["Rent"], kde=True)
plt.title("Distribution of Rent (Outliers Removed)")
plt.savefig("graph/rent_distribution.png")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(data=df_plot, x="Bed", y="Rent")
plt.title("Rent vs Beds")
plt.savefig("graph/rent_vs_bed.png")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(data=df_plot, x="Bath", y="Rent")
plt.title("Rent vs Bathrooms")
plt.savefig("graph/rent_vs_bath.png")
plt.show()

print("\nEDA Completed. Charts saved locally.")
