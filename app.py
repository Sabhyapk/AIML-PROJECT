# app.py

from flask import Flask, render_template, request
import pickle
import pandas as pd
from ml_model import MultiOutputModel  # ✅ Import it here too

app = Flask(__name__)

with open("model/symptom_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    age = int(request.form["age"])
    gender = request.form["gender"].lower()
    symptoms = request.form["symptoms"].lower().strip()
    gender_encoded = 0 if gender == "male" else 1

    input_df = pd.DataFrame([[age, gender_encoded, symptoms]],
                            columns=["age", "gender", "symptoms"])

    result = model.predict(input_df)

    return render_template("result.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)

