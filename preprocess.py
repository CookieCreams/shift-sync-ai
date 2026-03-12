import pandas as pd
import numpy as np

def clean_hospital_data(file_path):
    # 1. Chargement
    df = pd.read_csv(file_path)
    
    # 2. Remplacer les '?' par NaN (format standard pour les données manquantes)
    df.replace('?', np.nan, inplace=True)
    
    # 3. Supprimer les colonnes trop vides (ex: weight, medical_specialty, payer_code)
    # 'weight' est vide à 97%, on ne peut rien en faire.
    df.drop(['weight', 'medical_specialty', 'payer_code'], axis=1, inplace=True)
    
    # 4. Créer la cible BINAIRE (Le coeur du projet)
    # On veut prédire si le patient revient à MOINS de 30 jours (étiquette 1)
    df['target'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)
    
    # 5. Sélection des variables clés (Features) pour rester "Low Resource"
    # On choisit les colonnes qui ont le plus de sens clinique
    features = [
        'age', 'time_in_hospital', 'num_lab_procedures', 'num_procedures',
        'num_medications', 'number_outpatient', 'number_emergency', 
        'number_inpatient', 'number_diagnoses', 'gender'
    ]
    
    X = df[features].copy()
    y = df['target']
    
    # 6. Nettoyage rapide des types
    # Transformer l'âge "[70-80)" en un chiffre simple (75)
    def age_to_mid(age_str):
        if pd.isna(age_str): return 50
        nums = "".join([c for c in age_str if c.isdigit() or c == '-']).split('-')
        return (int(nums[0]) + int(nums[1])) / 2

    X['age'] = X['age'].apply(age_to_mid)
    
    # Transformer le genre en binaire
    X['gender'] = X['gender'].apply(lambda x: 1 if x == 'Female' else 0)
    
    return X, y

# Test rapide
if __name__ == "__main__":
    X, y = clean_hospital_data('diabetic_data.csv')
    print(f"Structure des données : {X.shape}")
    print(f"Nombre de réadmissions détectées : {y.sum()}")