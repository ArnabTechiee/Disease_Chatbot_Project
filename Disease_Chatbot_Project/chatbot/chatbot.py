import joblib
import pandas as pd
import numpy as np

# ✅ Load trained model & encoders
model = joblib.load("../models/xgboost_disease_model.pkl")
label_encoder = joblib.load("../models/label_encoder.pkl")
disease_mapping = joblib.load("../models/disease_mapping.pkl")

# ✅ Load model's feature names
model_features = model.feature_names_in_

# ✅ Define keyword mappings for flexible input handling
YES_WORDS = {"yes", "y", "i have", "i do", "yeah", "sure"}
NO_WORDS = {"no", "n", "i don't", "not really", "nah"}
NONE_WORDS = {"none", "nothing", "no history", "not applicable", "n/a"}

BP_OPTIONS = {"low": "Low", "normal": "Normal", "high": "High"}
CHOL_OPTIONS = {"low": "Low", "normal": "Normal", "high": "High"}

GENDER_OPTIONS = {
    "male": "Male", "m": "Male", "man": "Male", "boy": "Male", "guy": "Male", "gentleman": "Male",
    "female": "Female", "f": "Female", "woman": "Female", "girl": "Female", "lady": "Female",
    "other": "Other", "non-binary": "Other", "nb": "Other", "trans": "Other"
}

# ✅ Function to process user input
def process_input(user_response, valid_options=None):
    user_response = user_response.lower().strip()

    # Handle Yes/No cases
    if any(word in user_response for word in YES_WORDS):
        return "Yes"
    if any(word in user_response for word in NO_WORDS):
        return "No"
    if any(word in user_response for word in NONE_WORDS):
        return "None"

    # Handle categorical inputs (Blood Pressure, Cholesterol, Gender)
    if valid_options:
        for key, value in valid_options.items():
            if key in user_response:
                return value

    return user_response.capitalize()

# ✅ Chatbot Interaction
print("\n🤖 Welcome to the **AI Medical Assistant!** I’ll help assess your symptoms.\n")

# 📝 Collect User Info
name = input("👤 What is your name?\n➡️ ").strip().capitalize()
print(f"\n👋 Hello, {name}! Nice to meet you. Let's begin.\n")

age = input("📅 How old are you?\n➡️ ").strip()
print(f"\nGot it, {name}. You're {age} years old.\n")

gender = process_input(input("⚧ What is your gender? (Male/Female/Other)\n➡️ "), GENDER_OPTIONS)
history = process_input(input("\n📖 Do you have any medical history? (If none, type 'None')\n➡️ "))

# 🩸 Blood Pressure & Cholesterol Level
blood_pressure = process_input(input("\n🩸 What is your blood pressure? (Low/Normal/High)\n➡️ "), BP_OPTIONS)
cholesterol = process_input(input("\n🩺 What is your cholesterol level? (Low/Normal/High)\n➡️ "), CHOL_OPTIONS)

# 🩹 Symptom Check
print("\n💬 Now, let's check your symptoms. Just answer 'Yes' or 'No'.\n")

fever = process_input(input("🌡 Do you have a fever?\n➡️ "))
if fever == "Yes":
    print("😞 Sorry to hear that, fever can be really uncomfortable.")

cough = process_input(input("\n😷 Are you experiencing a cough?\n➡️ "))
if cough == "Yes":
    print("🤒 Oh no, coughing can be a sign of infection. Take care!")

fatigue = process_input(input("\n💤 Are you feeling fatigued or weak?\n➡️ "))
if fatigue == "Yes":
    print("😔 Fatigue can be tough. Make sure to rest well.")

breathing_difficulty = process_input(input("\n😮‍💨 Are you having trouble breathing?\n➡️ "))
if breathing_difficulty == "Yes":
    print("⚠️ Difficulty breathing is serious. Consider seeking medical help if it's severe.")

# ✅ Create user data dictionary
user_data = {
    "Age": int(age),
    "Gender_Male": 1 if gender == "Male" else 0,
    "Gender_Female": 1 if gender == "Female" else 0,  # ✅ Fix: Ensure "Gender_Female" exists
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

# ✅ Convert dictionary to DataFrame & align with model features
user_input = pd.DataFrame([user_data])

# ✅ **Fix: Ensure Missing Columns Are Added**
for col in model_features:
    if col not in user_input:
        user_input[col] = 0  # Add missing columns with default values

# ✅ Reorder columns to match the model
user_input = user_input[model_features]

# ✅ Predict disease
predicted_probs = model.predict_proba(user_input)[0]  # Get probability scores
max_index = np.argmax(predicted_probs)  # Get the highest probability disease
predicted_disease = disease_mapping.get(max_index, "Unknown Disease")
confidence = predicted_probs[max_index] * 100  # Confidence score

# 🎉 Display Results
print("\n🩺 Diagnosis Results:")
print(f"🔍 Predicted Disease: {predicted_disease}")
print(f"📊 Confidence Score: {confidence:.2f}%")

print(f"\n🤖 {name}, thank you for using the AI Medical Assistant! Stay healthy! 😊")
