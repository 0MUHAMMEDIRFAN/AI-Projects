# predict.py
from model import load_model

# Load trained model
model = load_model()

# New order input
order = {
    "food_item": "Pizza",
    "quantity": 40,
    "hour": 14,
    "day": "Monday"
}

prediction = model.predict_one(order)

if prediction:
    print(f"⏳ Estimated preparation time: {prediction:.2f} minutes")
else:
    print("🤖 Model not ready. Feed it with some data first.")
