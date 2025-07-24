# train_model.py

import pandas as pd
import pickle
import os
from ml_model import MultiOutputModel  # ✅ import here

df = pd.read_csv("AI_Symptom_Checker_Dataset.csv")
df.columns = df.columns.str.strip().str.lower()
df.dropna(subset=['age', 'gender', 'symptoms', 'predicted disease', 'confidence score (%)', 'severity'], inplace=True)
df['gender'] = df['gender'].str.lower().str.strip().map({'male': 0, 'female': 1})

X = df[['age', 'gender', 'symptoms']]
y = pd.DataFrame({
    'disease': df['predicted disease'],
    'severity': df['severity'],
    'confidence': df['confidence score (%)']
})

model = MultiOutputModel()
model.fit(X, y)

os.makedirs("model", exist_ok=True)
with open("model/symptom_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved!")
