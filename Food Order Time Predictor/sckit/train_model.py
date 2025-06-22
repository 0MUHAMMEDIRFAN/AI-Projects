import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load data
df = pd.read_csv('../data/training_data.csv')

# Convert categorical values
df = pd.get_dummies(df, columns=['food_item', 'day_of_week'])

# Features and target
X = df.drop(columns=['prep_time'])
y = df['prep_time']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, '../model/prep_time_model.pkl')
print("✅ Model trained and saved.")
