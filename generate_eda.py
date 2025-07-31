# generate_eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Créer le dossier pour les visualisations
if not os.path.exists('static/plots'):
    os.makedirs('static/plots')

# Charger les données
df = pd.read_csv('data/diabetes.csv')

# Remplacer les zéros non pertinents par NaN
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zeros] = df[cols_with_zeros].replace(0, pd.NA)
df[cols_with_zeros] = df[cols_with_zeros].fillna(df[cols_with_zeros].median()).infer_objects(copy=False)

# Statistiques descriptives (sauvegardées dans un fichier texte)
stats = df.describe().to_html()
with open('static/plots/descriptive_stats.html', 'w') as f:
    f.write(stats)

# Répartition de la variable cible
outcome_counts = df['Outcome'].value_counts(normalize=True).to_frame().to_html()
with open('static/plots/outcome_distribution.html', 'w') as f:
    f.write(outcome_counts)

# Histogrammes
df.hist(figsize=(12, 10))
plt.tight_layout()
plt.savefig('static/plots/histograms.png')
plt.close()

# Matrice de corrélation
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matrice de corrélation')
plt.savefig('static/plots/correlation_matrix.png')
plt.close()

# Boxplots
for col in cols_with_zeros:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Outcome', y=col, data=df)
    plt.title(f'Boxplot de {col} par Outcome')
    plt.savefig(f'static/plots/boxplot_{col}.png')
    plt.close()

print("Visualisations EDA sauvegardées dans 'static/plots/'.")