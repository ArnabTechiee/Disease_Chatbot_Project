import os
import faiss
import pandas as pd
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
import re

# Load dataset paths
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "..", "data")
model_path = os.path.join(current_dir, "..", "models")
faiss_index_path = os.path.join(model_path, "faiss_index.idx")

# Load dataset
file_path = os.path.join(data_path, "medquad.csv")
if not os.path.exists(file_path):
    raise FileNotFoundError(f"⚠ ERROR: File '{file_path}' not found! Place it in 'data/'.")

df = pd.read_csv(file_path, usecols=["question", "answer"])
if df.empty:
    raise ValueError(f"⚠ ERROR: File '{file_path}' is empty or corrupt.")

questions = df["question"].tolist()
answers = df["answer"].tolist()

# Load Sentence Transformer Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Faster FAISS loading (Precompute embeddings)
if os.path.exists(faiss_index_path):
    index = faiss.read_index(faiss_index_path)
else:
    print("🛠 Generating FAISS index for medical Q&A...")
    embeddings = embedding_model.encode(questions, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, faiss_index_path)

# Load trained XGBoost model & Label Encoder
disease_model = joblib.load(os.path.join(model_path, "xgboost_disease_model.pkl"))
label_encoder = joblib.load(os.path.join(model_path, "label_encoder.pkl"))

# Predefined Symptom List
SYMPTOM_LIST = [
    "fever", "cough", "fatigue", "sore throat", "runny nose", "headache",
    "chest pain", "difficulty breathing", "nausea", "vomiting", "diarrhea",
    "joint pain", "rash", "dizziness", "loss of taste", "loss of smell",
    "stomach pain", "sweating", "muscle pain", "sneezing"
]

def extract_symptoms(user_input):
    """ Extract symptoms using keyword matching """
    detected = [symptom for symptom in SYMPTOM_LIST if re.search(rf"\b{symptom}\b", user_input.lower())]
    return detected

def generate_faiss_query(disease, symptoms):
    """ Generate diverse question-based embeddings for better FAISS matching """
    treatment_queries = [
        f"How should {disease} be treated?",
        f"What medications are available for {disease}?",
        f"What are the common treatments for {disease}?",
        f"My symptoms include {', '.join(symptoms)}. What is the best treatment for {disease}?"
    ]

    # Average embeddings from multiple query styles for robustness
    query_embeddings = embedding_model.encode(treatment_queries)
    query_embedding = np.mean(query_embeddings, axis=0).reshape(1, -1)

    return query_embedding

def retrieve_answer(query_embedding):
    """ Search FAISS for the best answer """
    _, top_match = index.search(query_embedding, 1)
    if top_match[0][0] == -1:
        return None
    return answers[top_match[0][0]]

def medical_chatbot():
    """ Interactive AI Medical Chatbot with User Profile-Based Diagnosis """

    while True:
        print("\n🤖 Chatbot: Hi there! I'm your medical assistant. Let's start by understanding your health profile.")

        # Step 1: Collect User Profile
        user_name = ""
        while not user_name.strip():
            user_name = input("👤 You: What’s your name? ").strip().capitalize()
            if not user_name:
                print("🤖 Chatbot: Please provide a valid name.")

        print(f"\n🤖 Chatbot: Nice to meet you, {user_name}! Let's start by understanding your health profile.")
        while True:
            user_age = input("👤 You: How old are you? ").strip()
            if user_age.isdigit():
                user_age = int(user_age)
                break
            print("🤖 Chatbot: Please enter a valid number for your age.")

        while True:
            user_gender = input("👤 You: What is your gender? (Male/Female) ").strip().lower()
            if user_gender in ["male", "female"]:
                break
            print("🤖 Chatbot: Please enter either 'Male' or 'Female'.")

        user_medical_history = input("👤 You: Do you have any pre-existing conditions (e.g., diabetes, hypertension)? If none, type 'No'. ").strip().lower()

        print("\n🤖 Chatbot: Thanks for sharing your health profile! Now, let's talk about your symptoms.")

        detected_symptoms = []

        # Step 2: Collect Symptoms
        while True:
            user_input = input("👤 You: ").strip().lower()

            if user_input in ["exit", "quit", "stop", "bye"]:
                print("👋 Chatbot: Take care! Stay healthy. Exiting now.")
                return

            if user_input in ["no", "nothing else", "that's all"]:
                if not detected_symptoms:
                    print("\n🤖 Chatbot: I couldn't detect specific symptoms. Try describing them differently?")
                    return
                break

            new_symptoms = extract_symptoms(user_input)

            if new_symptoms:
                detected_symptoms.extend(new_symptoms)
                print(f"🤖 Chatbot: Hmm, I see {', '.join(new_symptoms)}. Anything else?")
            else:
                print("🤖 Chatbot: Could you describe your symptoms in more detail?")

        # ✅ Analyze Symptoms After User Stops Input
        print("\n🤖 Chatbot: Alright, let me analyze your symptoms...")

        feature_names = disease_model.get_booster().feature_names
        user_data = np.zeros(len(feature_names))

        for i, symptom in enumerate(feature_names):
            if symptom in detected_symptoms:
                user_data[i] = 1

        input_df = pd.DataFrame([user_data], columns=feature_names)

        prediction_proba = disease_model.predict_proba(input_df)[0]
        top_indices = np.argsort(prediction_proba)[::-1][:3]

        print("\n🩺 Chatbot: Based on your symptoms and health profile, here are the most likely conditions:")
        for rank, idx in enumerate(top_indices, start=1):
            disease_name = label_encoder.inverse_transform([idx])[0]
            confidence = prediction_proba[idx] * 100
            print(f"  {rank}. {disease_name} - {confidence:.2f}% confidence")

        # 🔄 Improved FAISS retrieval
        top_disease = label_encoder.inverse_transform([top_indices[0]])[0]
        query_embedding = generate_faiss_query(top_disease, detected_symptoms)
        treatment = retrieve_answer(query_embedding) or "🤖 No specific treatment found. Please consult a doctor."

        query_embedding = generate_faiss_query(top_disease, detected_symptoms)
        precautions = retrieve_answer(query_embedding) or "🤖 No specific precautions found. Please consult a doctor."

        print("\n📌 Treatment Advice:", treatment)
        print("⚠ Precautions You Should Take:", precautions)

        print(f"\n🩺 *Final Diagnosis:* Based on my analysis, you are most likely suffering from *{top_disease}*.")

        # Step 3: Restart or Exit
        if input("\n🤖 Chatbot: Would you like to start a new session? (Yes/No) ").strip().lower() != 'yes':
            print("\n👋 Chatbot: Thank you for using the medical assistant. Stay healthy! Exiting now.")
            break



