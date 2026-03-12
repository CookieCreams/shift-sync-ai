import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from preprocess import clean_hospital_data

# 1. Charger les données et le modèle
X, y = clean_hospital_data('diabetic_data.csv')
model = joblib.load('hospital_model.pkl')

# 2. Créer l'expliqueur SHAP (spécifique aux arbres comme Random Forest)
explainer = shap.TreeExplainer(model)
# On prend un échantillon pour que ce soit rapide
X_sample = X.sample(100, random_state=42)
shap_values = explainer.shap_values(X_sample)

# 3. Visualisation
print("📊 Génération de la vue d'ensemble des facteurs d'influence...")
# Si shap_values est une liste (cas du Random Forest), on prend l'indice 1 (réadmission)
sv = shap_values[1] if isinstance(shap_values, list) else shap_values

shap.summary_plot(sv, X_sample, show=False)
plt.savefig('shap_summary.png')
print("✅ Image sauvegardée : 'shap_summary.png'")