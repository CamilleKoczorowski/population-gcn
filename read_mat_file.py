import scipy.io as sio

# Chemin vers vos fichiers
file_abide = "data/ABIDE_aal/50003_aal_correlation.mat"
file_adhd = "data/ADHD_aal/10001_aal_correlation.mat"

def inspect_mat(filename):
    try:
        # Chargement
        mat_content = sio.loadmat(filename)
        
        # On regarde les clés (noms des variables)
        # On ignore les clés techniques (__header__, __version__, __globals__)
        variables = [key for key in mat_content.keys() if not key.startswith('__')]
        
        print(f"--- Analyse de {filename} ---")
        print(f"Variables trouvées : {variables}")
        
        # On inspecte la variable 'connectivity'
        if 'connectivity' in variables:
            data = mat_content['connectivity']
            print(f"Forme de la matrice : {data.shape}")
            print(f"Type de données : {data.dtype}")
            print(f"Aperçu (case 0,0) : {data[0,0]}")
        else:
            print("ATTENTION : Pas de clé 'connectivity' trouvée !")
            
    except Exception as e:
        print(f"Erreur de lecture : {e}")

# Lancer l'inspection
inspect_mat(file_abide)
print()
inspect_mat(file_adhd)