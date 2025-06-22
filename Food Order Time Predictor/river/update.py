# update.py
from model import load_model, save_model
import csv

# Load data from CSV
examples = []
actual_prep_times = []
with open("data/training_data.csv", newline="") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        example = {
            "food_item": row["food_item"],
            "quantity": int(row["quantity"]),
            "hour": int(row["hour"]),
            "day": row["day_of_week"]
        }
        examples.append(example)
        actual_prep_times.append(float(row["prep_time"]))

# Load model and learn
model = load_model()
for example, actual_prep_time in zip(examples, actual_prep_times):
    model.learn_one(example, actual_prep_time)
save_model(model)

print("✅ Model updated with new order data.")
