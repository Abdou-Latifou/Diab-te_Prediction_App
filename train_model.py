# train_model.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# Charger les données
df = pd.read_csv('data/diabetes.csv')

# Remplacer les zéros non pertinents par NaN
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zeros] = df[cols_with_zeros].replace(0, pd.NA)
df[cols_with_zeros] = df[cols_with_zeros].fillna(df[cols_with_zeros].median()).infer_objects(copy=False)

# Séparer les caractéristiques et la cible
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Vérifier les colonnes
print("Colonnes dans X pour l'entraînement:", X.columns.tolist())

# Diviser les données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Gérer le déséquilibre des classes avec SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Entraîner le modèle
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# Évaluer le modèle
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Sauvegarder le modèle
joblib.dump(model, 'diabetes_model.pkl')
print("Modèle sauvegardé sous 'diabetes_model.pkl'.")