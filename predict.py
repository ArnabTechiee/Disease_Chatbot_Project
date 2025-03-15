import joblib
import pandas as pd
import numpy as np
import sys
import json

model = joblib.load("../models/xgboost_disease_model.pkl")
disease_mapping = joblib.load("../models/disease_mapping.pkl")
model_features = model.feature_names_in_

def process_input(user_response, valid_options=None):
    YES_WORDS = {"yes", "y", "i have", "i do", "yeah", "sure"}
    NO_WORDS = {"no", "n", "i don't", "not really", "nah"}
    NONE_WORDS = {"none", "nothing", "no history", "not applicable", "n/a"}
    
    user_response = user_response.lower().strip()

    if any(word in user_response for word in YES_WORDS):
        return "Yes"
    if any(word in user_response for word in NO_WORDS):
        return "No"
    if any(word in user_response for word in NONE_WORDS):
        return "None"

    if valid_options:
        for key, value in valid_options.items():
            if key in user_response:
                return value

    return user_response.capitalize()

input_data = json.loads(sys.argv[1])

age = input_data.get("age", "")
gender = process_input(input_data.get("gender", ""), None)
history = process_input(input_data.get("history", ""), None)
blood_pressure = process_input(input_data.get("blood_pressure", ""), None)
cholesterol = process_input(input_data.get("cholesterol", ""), None)
fever = process_input(input_data.get("fever", ""), None)
cough = process_input(input_data.get("cough", ""), None)
fatigue = process_input(input_data.get("fatigue", ""), None)
breathing_difficulty = process_input(input_data.get("breathing_difficulty", ""), None)

user_data = {
    "Age": int(age),
    "Gender_Male": 1 if gender == "Male" else 0,
    "Gender_Female": 1 if gender == "Female" else 0,
    "Blood Pressure_Low": 1 if blood_pressure == "Low" else 0,
    "Blood Pressure_Normal": 1 if blood_pressure == "Normal" else 0,
    "Blood Pressure_High": 1 if blood_pressure == "High" else 0,
    "Cholesterol Level_Low": 1 if cholesterol == "Low" else 0,
    "Cholesterol Level_Normal": 1 if cholesterol == "Normal" else 0,
    "Cholesterol Level_High": 1 if cholesterol == "High" else 0,
    "Fever": 1 if fever == "Yes" else 0,
    "Cough": 1 if cough == "Yes" else 0,
    "Fatigue": 1 if fatigue == "Yes" else 0,
    "Difficulty Breathing": 1 if breathing_difficulty == "Yes" else 0
}

user_input = pd.DataFrame([user_data])

for col in model_features:
    if col not in user_input:
        user_input[col] = 0

user_input = user_input[model_features]

predicted_probs = model.predict_proba(user_input)[0]
max_index = np.argmax(predicted_probs)
predicted_disease = disease_mapping.get(max_index, "Unknown Disease")
confidence = predicted_probs[max_index] * 100

result = {
    "predicted_disease": predicted_disease,
    "confidence": round(confidence, 2)
}

print(json.dumps(result))
