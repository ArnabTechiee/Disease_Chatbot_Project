import pandas as pd
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

#  Load the dataset
df = pd.read_csv("data/Disease_symptom_and_patient_profile_dataset.csv")

#  Remove rare diseases (less than 2 instances)
disease_counts = df["Disease"].value_counts()
df = df[df["Disease"].isin(disease_counts[disease_counts > 1].index)]

#  Convert "Yes"/"No" to numerical values (Fixed FutureWarning)
symptom_cols = ["Fever", "Cough", "Fatigue", "Difficulty Breathing"]
df[symptom_cols] = df[symptom_cols].replace({"No": 0, "Yes": 1}).infer_objects(copy=False).astype(int)

#  Prepare feature and target columns
X = df.drop(columns=["Disease", "Outcome Variable"])
y = df["Disease"]

#  Convert categorical values into numbers
X = pd.get_dummies(X, columns=["Gender", "Blood Pressure", "Cholesterol Level"], drop_first=True)

#  Encode disease names into numerical labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

#  Print class distribution before SMOTE
print(" Class distribution before SMOTE:\n", pd.Series(y_encoded).value_counts())

#  Apply SMOTE with k_neighbors=1 (Fixed Error)
try:
    smote = SMOTE(sampling_strategy="auto", k_neighbors=1, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y_encoded)
    print(" SMOTE applied successfully!")
except ValueError as e:
    print("⚠ Skipping SMOTE due to insufficient samples:", e)
    X_resampled, y_resampled = X, y_encoded  # Use original dataset if SMOTE fails

#  Print class distribution after applying SMOTE
print(" Class distribution after SMOTE:\n", pd.Series(y_resampled).value_counts())

#  Split dataset into training & testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

#  Train the XGBoost model with optimized parameters
model = XGBClassifier(
    eval_metric="mlogloss",
    objective="multi:softmax",
    num_class=len(label_encoder.classes_),
    learning_rate=0.05,  # Lower learning rate for better convergence
    max_depth=8,  # Deeper trees for better learning
    n_estimators=500,  # More estimators for improved accuracy
    subsample=0.9,  # More data used for training
    colsample_bytree=0.9,  # Better feature selection
)

model.fit(X_train, y_train)

#  Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f" Model Training Complete! Accuracy: {accuracy * 100:.2f}%")

#  Save the trained model and label encoder
joblib.dump(model, "models/xgboost_disease_model.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")  # Save label encoder for chatbot use
print(" Model saved as 'models/xgboost_disease_model.pkl'")
print(" Label encoder saved as 'models/label_encoder.pkl'")
