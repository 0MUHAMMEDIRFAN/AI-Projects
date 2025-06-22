import joblib

# Load model
model = joblib.load('marks_model.pkl')

# Input new student data
new_data = [[1, 70, 15]]  # 6 hours studied, 7 hours sleep, 85% attendance

# Predict
predicted_marks = model.predict(new_data)
print(f"Predicted Marks: {predicted_marks[0]:.2f}")
