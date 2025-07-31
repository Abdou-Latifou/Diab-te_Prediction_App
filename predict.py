# predict.py
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os

# Créer un dossier pour sauvegarder les visualisations
if not os.path.exists('plots'):
    os.makedirs('plots')

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

# Faire des prédictions de probabilité
proba = model.predict_proba(X)[:, 1]  # Probabilité d'être diabétique

# Ajouter les probabilités au DataFrame
df['Risk_Probability'] = proba

# Afficher les premières lignes avec les probabilités
print("Exemple de prédictions avec probabilité de risque :")
print(df[['Outcome', 'Risk_Probability']].head())

# Visualisation SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Vérifier la forme des shap_values et de X
print("Forme de X:", X.shape)
print("Forme de shap_values:", len(shap_values) if isinstance(shap_values, list) else shap_values.shape)

# Si shap_values est une liste (pour classification binaire), utiliser shap_values[1] pour la classe positive
if isinstance(shap_values, list):
    shap.summary_plot(shap_values[1], X, show=False)
else:
    shap.summary_plot(shap_values, X, show=False)

plt.savefig('plots/shap_summary.png')
plt.close()

print("Interprétation SHAP sauvegardée dans 'plots/shap_summary.png'.")