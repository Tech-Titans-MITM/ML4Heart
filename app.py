from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('heart_disease_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Feature order
features = [
    'age', 'gender', 'cp', 'trestbps', 'chol',
    'fbs', 'restecg', 'thalach', 'exang',
    'oldpeak', 'slope', 'ca', 'thal'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_values = [float(request.form[feature]) for feature in features]
    input_df = pd.DataFrame([input_values], columns=features)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    result = 'High Risk (Heart Disease Likely)' if prediction == 1 else 'Low Risk (No Heart Disease)'
    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
