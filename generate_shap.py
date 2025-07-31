# generate_shap.py
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os

# Créer le dossier pour les visualisations
if not os.path.exists('static/plots'):
    os.makedirs('static/plots')

# Charger le modèle
model = joblib.load('diabetes_model.pkl')

# Charger les données
df = pd.read_csv('data/diabetes.csv')

# Remplacer les zéros non pertinents par NaN
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zeros] = df[cols_with_zeros].replace(0, pd.NA)
df[cols_with_zeros] = df[cols_with_zeros].fillna(df[cols_with_zeros].median()).infer_objects(copy=False)

# Séparer les caractéristiques
X = df.drop('Outcome', axis=1)

# Vérifier les colonnes
print("Colonnes dans X:", X.columns.tolist())

# Générer les valeurs SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Sauvegarder la visualisation SHAP
if isinstance(shap_values, list):
    shap.summary_plot(shap_values[1], X, show=False)
else:
    shap.summary_plot(shap_values, X, show=False)
plt.savefig('static/plots/shap_summary.png')
plt.close()

# Sauvegarder les probabilités de risque
proba = model.predict_proba(X)[:, 1]
df['Risk_Probability'] = proba
df[['Outcome', 'Risk_Probability']].to_html('static/plots/risk_probabilities.html')

print("Visualisation SHAP et probabilités sauvegardées dans 'static/plots/'.")