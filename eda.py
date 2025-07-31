# eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Créer un dossier pour sauvegarder les visualisations
if not os.path.exists('plots'):
    os.makedirs('plots')

# Charger les données
df = pd.read_csv('data/diabetes.csv')

# Remplacer les zéros non pertinents par NaN
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zeros] = df[cols_with_zeros].replace(0, pd.NA)

# Imputer les valeurs manquantes avec la médiane
df[cols_with_zeros] = df[cols_with_zeros].fillna(df[cols_with_zeros].median())

# Statistiques descriptives
print("Statistiques descriptives :")
print(df.describe())

# Distribution de la variable cible (Outcome)
print("\nRépartition de la variable cible (Outcome) :")
print(df['Outcome'].value_counts(normalize=True))

# Histogrammes des variables
df.hist(figsize=(12, 10))
plt.tight_layout()
plt.savefig('plots/histograms.png')
plt.close()

# Matrice de corrélation
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matrice de corrélation')
plt.savefig('plots/correlation_matrix.png')
plt.close()

# Boxplots pour comparer les variables par Outcome
for col in cols_with_zeros:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Outcome', y=col, data=df)
    plt.title(f'Boxplot de {col} par Outcome')
    plt.savefig(f'plots/boxplot_{col}.png')
    plt.close()

print("Visualisations sauvegardées dans le dossier 'plots'.")