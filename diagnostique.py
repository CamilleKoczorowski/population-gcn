import scipy.io as sio
import pandas as pd
import os
import glob

# Adaptez ce chemin si vos .mat sont dans un sous-dossier
data_path = "data/ADHD_aal" 

# 1. Inspection du CSV
print("--- INFOS CSV ---")
try:
    # On cherche le fichier csv
    csv_files = glob.glob(os.path.join(data_path, "*.csv"))
    if csv_files:
        df = pd.read_csv(csv_files[0])
        print("Colonnes trouvées :", df.columns.tolist())
        print("Exemple de ligne 1 :", df.iloc[0].to_dict())
    else:
        print("Aucun CSV trouvé dans", data_path)
except Exception as e:
    print("Erreur CSV:", e)

# 2. Inspection d'un .mat
print("\n--- INFOS MAT ---")
mat_files = glob.glob(os.path.join(data_path, "*.mat")) # ou "*/*.mat" si sous-dossiers
if mat_files:
    print(f"Fichier testé : {mat_files[0]}")
    mat = sio.loadmat(mat_files[0])
    # On filtre les clés techniques (__header__, etc)
    keys = [k for k in mat.keys() if not k.startswith('__')]
    print("Clés disponibles dans le .mat :", keys)
    print("Dimension de la première matrice :", mat[keys[0]].shape)
else:
    print("Aucun fichier .mat trouvé.")