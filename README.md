<<<<<<< HEAD
# Diab-te_Prediction_App
Web app for diabetes prediction using ML, built with Flask for AIMS-SENEGAL 2025. Features individual prediction, exploratory data analysis, and future risk assessment with SHAP. Modern, responsive UI with prevention tips.  Tech: Python, Flask, Pandas, Scikit-learn, SHAP, HTML, CSS, JS.
=======
# Diabetes Prediction App

## Prérequis
- Python 3.8+
- Visual Studio Code
- Bibliothèques : voir `requirements.txt`

## Installation
1. Créez un environnement virtuel : `python -m venv venv`
2. Activez l'environnement :
   - Windows : `venv\Scripts\activate`
   - Linux/Mac : `source venv/bin/activate`
3. Installez les dépendances : `pip install -r requirements.txt`
4. Placez le fichier `diabetes.csv` dans le dossier `data/`.

## Exécution
1. Entraînement du modèle : `python train_model.py`
   - Crée `diabetes_model.pkl`.
2. Générer les visualisations EDA : `python generate_eda.py`
   - Sauvegarde les visualisations dans `static/plots/`.
3. Générer la visualisation SHAP et les probabilités : `python generate_shap.py`
   - Sauvegarde `shap_summary.png` et `risk_probabilities.html` dans `static/plots/`.
4. Lancer l'application web : `python app.py`
   - Accédez à `http://127.0.0.1:5000` dans un navigateur.

## Fonctionnalités
- **Accueil** : Page d'introduction avec des liens vers les autres fonctionnalités.
- **Analyse exploratoire (EDA)** : Affiche les statistiques descriptives, la répartition de la variable cible, et des visualisations (histogrammes, matrice de corrélation, boxplots).
- **Prédiction individuelle** : Soumettez les données d'un patient pour obtenir une prédiction (diabétique ou non) et une probabilité de risque.
- **Risques futurs** : Affiche les probabilités de risque pour l'ensemble des données et une visualisation SHAP pour l'interprétation.

## Structure
- `data/` : Contient le jeu de données.
- `static/css/` : Styles CSS pour l'interface web.
- `static/plots/` : Visualisations générées.
- `templates/` : Modèles HTML pour Flask.
>>>>>>> 361b852 (Initial commit: Diabetes Prediction App with Flask, ML, and responsive UI)
