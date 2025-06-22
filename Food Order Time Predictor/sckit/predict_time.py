import joblib
import pandas as pd
from datetime import datetime

# Input order
order = {
    "food_item": "Pizza",
    "quantity": 2,
    "hour": datetime.now().hour,
    "day_of_week": datetime.now().strftime("%A")
}

# Load model
model = joblib.load('../model/prep_time_model.pkl')
data = pd.DataFrame([order])
print(data,"🔍 Preparing data for prediction...")
# Dummy variables like in training
all_foods = ['Pizza','Burger', 'Fries', 'Pasta', 'Salad', 'Sushi']
all_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

for item in all_foods:
    data[f"food_item_{item}"] = 1 if order["food_item"] == item else 0

for day in all_days:
    data[f"day_of_week_{day}"] = 1 if order["day_of_week"] == day else 0

data = data.drop(columns=["food_item", "day_of_week"])
print(data, "🔍 Data after dummy variable conversion...")
# Predict
predicted_time = model.predict(data)[0]
print(f"⏳ Estimated preparation time: {predicted_time:.2f} minutes")
