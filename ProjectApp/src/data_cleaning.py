import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

input_filepath = os.getenv("raw_csv_path")
output_filepath = os.getenv("cleaned_csv_path")

# Step 1: Load the dataset
df = pd.read_csv(input_filepath)

# Step 2: Drop rows with too many missing values (optional but safe)
df.dropna(thresh=15, inplace=True)  # Keep rows with at least 15 non-NaN values

# Step 3: Fill remaining missing values with "Unknown" or appropriate placeholder
df.fillna("Unknown", inplace=True)

# Step 4: Rename columns for ease of use (short and Pythonic)
df.columns = [
    "age_group", "gender", "city", "toilet_cleanliness", "toilet_safety",
    "toilet_features", "service_use", "service_use_freq", "transport_satisfaction",
    "transport_suggestions", "park_visiting", "park_visit_freq", "park_amenities",
    "park_issues", "transport_safety", "park_suggestions", "library_satisfaction",
    "library_visit_freq", "local_service_satisfaction", "library_suggestions",
    "local_service_suggestions"
]

# Step 5: Preview cleaned dataset
print("🧹 Cleaned Data Sample:")
print(df.head())

# Step 6: Save the cleaned data for next phase (optional)
df.to_csv(output_filepath, index=False)