# app.py
from flask import Flask, request, render_template
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Charger le modèle
model = joblib.load('diabetes_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/eda')
def eda():
    # Liste des fichiers de visualisations
    plot_files = [
        'histograms.png',
        'correlation_matrix.png',
    ] + [f'boxplot_{col}.png' for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']]
    return render_template('eda.html', plot_files=plot_files)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Récupérer les données du formulaire
            features = [
                float(request.form['pregnancies']),
                float(request.form['glucose']),
                float(request.form['blood_pressure']),
                float(request.form['skin_thickness']),
                float(request.form['insulin']),
                float(request.form['bmi']),
                float(request.form['dpf']),
                float(request.form['age'])
            ]
            # Créer un DataFrame
            columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
            input_data = pd.DataFrame([features], columns=columns)
            # Faire la prédiction
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]
            result = 'Diabétique' if prediction == 1 else 'Non diabétique'
            risk = f'Probabilité de risque : {probability:.2%}'
            return render_template('predict.html', prediction=result, probability=risk)
        except Exception as e:
            return render_template('predict.html', error=f"Erreur : {str(e)}")
    return render_template('predict.html')

@app.route('/risk')
def risk():
    return render_template('risk.html')

if __name__ == '__main__':
    app.run(debug=True)