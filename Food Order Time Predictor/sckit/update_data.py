import pandas as pd
from datetime import datetime

# Add this after food is done
actual_data = {
    "food_item": "Pizza",
    "quantity": 2,
    "hour": 13,  # Example: order hour
    "day_of_week": "Monday",
    "prep_time": 14.0  # Actual time it took
}

# Append to CSV
df = pd.read_csv('../data/training_data.csv')
df = df.append(actual_data, ignore_index=True)
df.to_csv('../data/training_data.csv', index=False)

print("📦 Actual data saved. Run train_model.py to re-train the model.")
