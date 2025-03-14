import pandas as pd
import numpy as np
import faiss
import pickle
from sklearn.preprocessing import LabelEncoder

# ✅ Load the dataset
symptom_data = pd.read_csv("../data/Disease_symptom_and_patient_profile_dataset.csv")

# ✅ Convert Yes/No symptoms to numerical values
binary_cols = ["Fever", "Cough", "Fatigue", "Difficulty Breathing"]
for col in binary_cols:
    symptom_data[col] = symptom_data[col].map({"Yes": 1, "No": 0})

# ✅ Encode categorical columns (Gender)
le = LabelEncoder()
symptom_data["Gender"] = le.fit_transform(symptom_data["Gender"])

# ✅ Combine symptoms into a single vector for FAISS
symptom_vectors = symptom_data[binary_cols + ["Age", "Gender"]].values.astype(np.float32)

# 🔥 **Fix for FAISS Error: Convert to C-contiguous array**
symptom_vectors = np.ascontiguousarray(symptom_vectors, dtype=np.float32)

# ✅ Normalize FAISS vectors
faiss.normalize_L2(symptom_vectors)

# ✅ Create FAISS index
d = symptom_vectors.shape[1]
faiss_index = faiss.IndexFlatL2(d)
faiss_index.add(symptom_vectors)

# ✅ Save FAISS index
faiss.write_index(faiss_index, "../models/faiss_index.idx")

# ✅ Save encoders
with open("../models/disease_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("✅ Data preprocessing complete. FAISS index & encoders saved successfully!")
