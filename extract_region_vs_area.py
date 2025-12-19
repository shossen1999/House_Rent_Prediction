# extract_region_mapping.py
import pandas as pd
import json

df = pd.read_csv("data/cleaned_data.csv")

region_map = (
    df.groupby("Region")["SubArea"]
      .unique()
      .apply(list)
      .to_dict()
)

with open("region_subarea.json", "w") as f:
    json.dump(region_map, f, indent=4)

print("Region → SubArea mapping saved to region_subarea.json")
