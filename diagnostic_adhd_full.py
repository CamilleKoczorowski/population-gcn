import pandas as pd
import numpy as np
import os
import scipy.io as sio
import glob

# --- CONFIGURATION ---
# Assure-toi que ce chemin pointe bien vers ton dossier de données ADHD
data_folder = 'data/ADHD_aal'
csv_file_name = 'Phenotypic_V1_0b_preprocessed1.csv' 
#/Users/camillekoczo/Desktop/MVA/Semestre_1/GEOMETRIC DATA ANALYSIS/population-gcn/data/ADHD_aal/Phenotypic_V1_0b_preprocessed1.csv

def analyze_full_dataset():
    print("="*50)
    print("RAPPORT DE DIAGNOSTIC - DATASET ADHD-200")
    print("="*50)

    # 1. Chargement du CSV
    csv_path = os.path.join(data_folder, csv_file_name)
    if not os.path.exists(csv_path):
        print(f"ERREUR CRITIQUE : Le fichier {csv_path} est introuvable.")
        return

    df = pd.read_csv(csv_path)
    print(f"\n[1] DIMENSIONS DU CSV")
    print(f"Nombre de lignes (sujets) : {df.shape[0]}")
    print(f"Nombre de colonnes : {df.shape[1]}")
    print("-" * 30)
    print("Liste des colonnes :")
    print(df.columns.tolist())

    # 2. Analyse des Colonnes Clés (Phénotypes)
    print(f"\n[2] STATISTIQUES DESCRIPTIVES (Colonnes Clés)")
    
    # Analyse SITE
    if 'site' in df.columns:
        print("\n--- Répartition par SITE ---")
        print(df['site'].value_counts(dropna=False))
    
    # Analyse SEXE
    if 'sex' in df.columns:
        print("\n--- Répartition par SEXE ---")
        print(df['sex'].value_counts(dropna=False))
        
    # Analyse AGE
    if 'age' in df.columns:
        print("\n--- Statistiques AGE ---")
        print(df['age'].describe())

    # Analyse DIAGNOSTIC (La Target)
    # On cherche les colonnes potentielles pour le label
    diag_cols = [c for c in df.columns if 'adhd' in c.lower() or 'dx' in c.lower() or 'tdc' in c.lower()]
    print(f"\n--- Colonnes Diagnostic trouvées : {diag_cols} ---")
    
    if 'adhd' in df.columns:
        print("Distribution de la colonne 'adhd' :")
        print(df['adhd'].value_counts(dropna=False))
    
    # 3. Analyse des Données Matricielles (.mat)
    print(f"\n[3] ANALYSE DES FICHIERS .MAT (Connectivité)")
    mat_files = glob.glob(os.path.join(data_folder, '*.mat'))
    # Recherche aussi dans les sous-dossiers si jamais ils sont rangés par ID
    if len(mat_files) == 0:
        mat_files = glob.glob(os.path.join(data_folder, '*', '*.mat'))
    
    print(f"Nombre de fichiers .mat trouvés : {len(mat_files)}")

    if len(mat_files) > 0:
        # Inspection du premier fichier pour voir la structure
        sample_mat = mat_files[0]
        try:
            mat_content = sio.loadmat(sample_mat)
            keys = [k for k in mat_content.keys() if not k.startswith('__')]
            print(f"\nStructure du fichier '{os.path.basename(sample_mat)}' :")
            print(f"Clés disponibles : {keys}")
            
            # Vérification de la forme de la matrice
            for key in keys:
                if isinstance(mat_content[key], np.ndarray):
                    print(f"Dimensions de '{key}' : {mat_content[key].shape}")
                    
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier .mat : {e}")
            
        # 4. Vérification de la Correspondance (Matching)
        print(f"\n[4] VERIFICATION DE CORRESPONDANCE (CSV vs MAT)")
        if 'Subject' in df.columns:
            # On convertit les IDs du CSV en string pour comparer
            csv_ids = set(df['Subject'].astype(str))
            
            # On extrait les IDs des noms de fichiers (ex: "0010042_ho_correlation.mat" -> "0010042")
            # Cette logique dépend du nommage exact, on essaie de trouver l'ID du csv dans le nom du fichier
            files_ids = []
            for f in mat_files:
                fname = os.path.basename(f)
                for cid in csv_ids:
                    if cid in fname:
                        files_ids.append(cid)
                        break
            
            matched = len(set(files_ids))
            print(f"Sujets du CSV trouvés en .mat : {matched} / {len(csv_ids)}")
            
            if matched < len(csv_ids):
                print("ATTENTION : Certains sujets du CSV n'ont pas de fichier .mat correspondant.")
        else:
            print("Impossible de faire le matching : Colonne 'Subject' introuvable dans le CSV.")

    print("\n" + "="*50)
    print("FIN DU RAPPORT")
    print("="*50)

if __name__ == "__main__":
    analyze_full_dataset()