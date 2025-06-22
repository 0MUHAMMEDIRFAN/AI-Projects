import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load data
df = pd.read_csv("students.csv")
X = df[['Hours_Studied', 'Sleep_Hours', 'Attendance']]
y = df['Marks']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, 'marks_model.pkl')
print("Model trained and saved.")
