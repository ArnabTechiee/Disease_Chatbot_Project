import pandas as pd
import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# ✅ Load dataset
df = pd.read_csv("../data/Disease_symptom_and_patient_profile_dataset.csv")

# ✅ Remove rare diseases (less than 2 instances)
disease_counts = df["Disease"].value_counts()
df = df[df["Disease"].isin(disease_counts[disease_counts > 1].index)]

# ✅ Convert categorical "Yes"/"No" to numeric
symptom_cols = ["Fever", "Cough", "Fatigue", "Difficulty Breathing"]
df[symptom_cols] = df[symptom_cols].replace({"No": 0, "Yes": 1}).astype(int)

# ✅ Prepare features (X) and target (y)
X = df.drop(columns=["Disease", "Outcome Variable"])
y = df["Disease"]

# ✅ Apply One-Hot Encoding to categorical features
X = pd.get_dummies(X, columns=["Gender", "Blood Pressure", "Cholesterol Level"], drop_first=False)

# ✅ Ensure all expected columns exist (Fixing Missing Features)
expected_columns = [
    "Gender_Female", "Gender_Male",
    "Blood Pressure_Low", "Blood Pressure_Normal", "Blood Pressure_High",
    "Cholesterol Level_Low", "Cholesterol Level_Normal", "Cholesterol Level_High"
]
for col in expected_columns:
    if col not in X.columns:
        X[col] = 0  # Add missing columns with default 0

# ✅ Encode disease labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# ✅ Apply SMOTE
try:
    smote = SMOTE(k_neighbors=1, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y_encoded)
    print("✅ SMOTE applied successfully!")
except ValueError:
    print("⚠️ SMOTE not applied due to rare diseases.")
    X_resampled, y_resampled = X, y_encoded

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# ✅ Train XGBoost model
model = XGBClassifier(
    eval_metric="mlogloss",
    objective="multi:softmax",
    num_class=len(label_encoder.classes_),
    learning_rate=0.05,
    max_depth=8,
    n_estimators=500,
    subsample=0.9,
    colsample_bytree=0.9
)
model.fit(X_train, y_train)

# ✅ Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# ✅ Save trained model & encoder
joblib.dump(model, "../models/xgboost_disease_model.pkl")
joblib.dump(label_encoder, "../models/label_encoder.pkl")

# ✅ Print final accuracy
print(f"✅ Model Training Complete! Accuracy: {accuracy * 100:.2f}%")
print("✅ Model and encoder saved successfully.")

# ✅ Display Correct Disease Mapping
disease_mapping = dict(enumerate(label_encoder.classes_))
print("\n🔍 Correct Disease Mapping (Encoded Number → Disease Name):")
for index, disease in disease_mapping.items():
    print(f"{index}: {disease}")

# ✅ Save disease mapping
joblib.dump(disease_mapping, "../models/disease_mapping.pkl")
