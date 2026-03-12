from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib # Pour sauvegarder le modèle
from preprocess import clean_hospital_data # On réutilise ton script précédent

# 1. Préparation des données
print("📥 Chargement et nettoyage des données...")

X, y = clean_hospital_data('diabetic_data.csv')

# 2. Division Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Création du Random Forest
# n_estimators=100 signifie qu'on crée 100 arbres de décision qui vont voter
print("⚙️ Entraînement du Random Forest (100 arbres)...")
model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# 4. Évaluation
y_pred = model.predict(X_test)
print("\n--- RÉSULTATS DU MODÈLE ---")
print(f"Précision globale : {accuracy_score(y_test, y_pred):.2%}")
print("\nDétails par catégorie :")
print(classification_report(y_test, y_pred))

# 5. Sauvegarde pour l'interface Streamlit
joblib.dump(model, 'hospital_model.pkl')
print("\n✅ Modèle sauvegardé sous 'hospital_model.pkl'")