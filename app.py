import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Configuration
st.set_page_config(page_title="Shift-Sync AI", page_icon="🏥")

# 1. Charger le modèle
model = joblib.load('hospital_model.pkl')

st.title("🏥 Shift-Sync : Aide à la Décision")
st.write("Saisissez les données du patient pour évaluer le risque de réadmission.")

# 2. Formulaire de saisie
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Âge du patient", 0, 100, 65)
        inpatient = st.number_input("Nombre d'hospitalisations (an passé)", 0, 20, 1)
        emergency = st.number_input("Nombre de passages aux urgences (an passé)", 0, 20, 0)
        time_hosp = st.slider("Jours d'hospitalisation actuels", 1, 14, 3)
    
    with col2:
        meds = st.number_input("Nombre de médicaments", 1, 50, 10)
        diags = st.number_input("Nombre de diagnostics", 1, 15, 5)
        labs = st.number_input("Nombre d'analyses labo effectuées", 1, 100, 30)
        gender = st.selectbox("Genre", ["Femme", "Homme"])

# 3. Prédiction
if st.button("🚀 Évaluer le Risque"):
    # Transformer les entrées en format pour le modèle
    # L'ordre doit être : age, time_in_hospital, num_lab_procedures, num_procedures, 
    # num_medications, number_outpatient, number_emergency, number_inpatient, number_diagnoses, gender
    # (Note : on met 0 pour outpatient et 0 pour num_procedures par défaut ici)
    features = np.array([[age, time_hosp, labs, 0, meds, 0, emergency, inpatient, diags, 1 if gender=="Femme" else 0]])
    
    proba = model.predict_proba(features)[0][1]
    
    st.divider()
    
    # Affichage du score
    if proba > 0.6:
        st.error(f"### 🚩 RISQUE ÉLEVÉ : {proba:.1%}")
        st.write("**Recommandation :** Ne pas autoriser la sortie sans un plan de suivi infirmier à domicile.")
    elif proba > 0.35:
        st.warning(f"### ⚠️ RISQUE MODÉRÉ : {proba:.1%}")
        st.write("**Recommandation :** Programmer un appel de suivi à 48h.")
    else:
        st.success(f"### ✅ RISQUE FAIBLE : {proba:.1%}")
        st.write("**Recommandation :** Sortie standard autorisée.")

    st.info("💡 Ce score est basé sur l'importance historique des hospitalisations précédentes et la complexité médicamenteuse.")