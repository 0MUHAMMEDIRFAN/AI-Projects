# model.py
import os
import pickle
from river import linear_model, preprocessing, feature_extraction, compose

MODEL_PATH = '../model/food_model.pkl'

# Define the model pipeline
def build_model():
    return (
        preprocessing.OneHotEncoder() |      # Moved from feature_extraction
        preprocessing.StandardScaler() |
        linear_model.LinearRegression()
    )

# Save model
def save_model(model):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)  # ✅ Create folder if missing
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

# Load model
def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    return build_model()
