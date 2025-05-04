from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('heart_disease_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Feature order (match dataset column names exactly)
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
    # Get user input and convert to float
    input_values = [float(request.form['gender']) if f == 'sex' else float(request.form[f]) for f in features]
    
    # Create DataFrame
    input_df = pd.DataFrame([input_values], columns=features)
    
    # Scale input
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = int(model.predict(input_scaled)[0])  # Ensure it's int for HTML logic
    
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
