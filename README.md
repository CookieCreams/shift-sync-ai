# Shift-Sync-AI

Shift-Sync est une solution d'IA "Low-Resource" conçue pour prédire le risque de réadmission à 30 jours des patients diabétiques. Ce projet démontre l'intégration du Machine Learning dans un flux de travail clinique, en mettant l'accent sur l'explicabilité (XAI) et la sécurité patient.

## Objectif du Projet

Le but est d'identifier les patients fragiles avant leur sortie de l'hôpital. Une réadmission précoce (<30 jours) est souvent synonyme de complication médicale évitable et de coûts élevés pour le système de santé. Utilisation de SHAP pour extraire les facteurs d'influence et garantir qu'aucune décision n'est une "boîte noire".

## Structure du Projet

preprocess.py: Nettoyage des données, gestion des valeurs manquantes et feature engineering.

train_model.py: Entraînement du Random Forest avec gestion du déséquilibre des classes.

explain.py: Génération des graphiques de forces SHAP.

app.py: Interface utilisateur interactive.

## Installation

streamlit run app.py

## Analyse des Résultats

Le projet a permis de mettre en évidence les prédicteurs critiques de réadmission.

number_inpatient : L'historique des hospitalisations passées est le facteur de risque n°1.

number_diagnoses : La comorbidité augmente significativement la probabilité de retour.

Le projet a révélé un biais de survie dans les données, où le risque statistique stagne pour les tranches d'âge extrêmes (90+), soulignant la nécessité d'une expertise humaine en complément de l'IA.