import pandas as pd

def load_disease_data(filepath):
    """Loads and preprocesses the disease dataset."""
    df = pd.read_csv(filepath)

    # Encode categorical features
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df['Blood Pressure'] = df['Blood Pressure'].map({'Low': 0, 'Normal': 1, 'High': 2})
    df['Cholesterol Level'] = df['Cholesterol Level'].map({'Low': 0, 'Normal': 1, 'High': 2})

    # Convert "Yes"/"No" to 1/0 for symptoms
    symptom_cols = [col for col in df.columns if col not in ['Disease', 'Outcome Variable', 'Age', 'Gender', 'Blood Pressure', 'Cholesterol Level']]
    df[symptom_cols] = df[symptom_cols].applymap(lambda x: 1 if x == 'Yes' else 0)

    # Convert "Positive"/"Negative" to 1/0
    df['Outcome Variable'] = df['Outcome Variable'].map({'Positive': 1, 'Negative': 0})

    return df
