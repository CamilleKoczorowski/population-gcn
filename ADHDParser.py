import os
import csv
import numpy as np
import scipy.io as sio
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import RFE

# --- CONFIGURATION ---
root_folder = os.getcwd()
data_folder = os.path.join(root_folder, 'data', 'ADHD_aal')

def get_ids(num_subjects=None):
    """ Récupère les IDs depuis les fichiers .mat """
    try:
        files = [f for f in os.listdir(data_folder) if f.endswith('_aal_correlation.mat')]
    except FileNotFoundError:
        print(f"ERREUR: Dossier {data_folder} introuvable.")
        return np.array([])

    # Nettoyage : "10099_aal_correlation.mat" -> "10099"
    subject_IDs = [f.replace('_aal_correlation.mat', '') for f in files]
    
    if num_subjects is not None:
        subject_IDs = subject_IDs[:num_subjects]
    return np.array(subject_IDs)

def get_subject_score(subject_list, score):
    """ Récupère les scores depuis le CSV avec la bonne colonne ID """
    key_map = {
        'DX_GROUP': 'DX',
        'SITE_ID': 'Site',
        'SEX': 'Gender',
        'AGE_AT_SCAN': 'Age'
    }
    
    target_col = key_map.get(score, score)
    scores_dict = {}
    
    # Recherche du CSV
    csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    if not csv_files:
        print("ERREUR: Pas de CSV trouvé.")
        return {}
        
    csv_path = os.path.join(data_folder, csv_files[0])
    
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # --- CORRECTION ICI : Utilisation de 'ScanDir ID' ---
            if 'ScanDir ID' in row:
                sub_id = row['ScanDir ID']
            elif 'Subject' in row:
                sub_id = row['Subject']
            else:
                continue

            # On vérifie si ce sujet est demandé
            if str(sub_id) in subject_list:
                try:
                    val = row[target_col]
                    
                    # Conversion Diagnostic pour le GCN
                    if score == 'DX_GROUP':
                        val = float(val) 
                        val = int(val)
                        # 0 -> 2 (Control/Sain)
                        # 1,2,3 -> 1 (Malade)
                        if val == 0:
                            val = 2 
                        else:
                            val = 1 
                            
                    scores_dict[str(sub_id)] = val
                except ValueError:
                    pass
    return scores_dict

def get_networks(subject_list, kind, atlas_name="aal", variable='connectivity'):
    """ Charge les matrices .mat """
    all_networks = []
    for subject in subject_list:
        # Nom de fichier exact basé sur ton diagnostic
        fl = os.path.join(data_folder, f"{subject}_aal_correlation.mat")
        
        try:
            # Clé 'connectivity' confirmée par ton diagnostic
            matrix = sio.loadmat(fl)['connectivity']
            all_networks.append(matrix)
        except Exception as e:
            print(f"Erreur chargement {subject}: {e}")
            all_networks.append(np.zeros((116, 116)))

    # Vectorisation (Triangle supérieur)
    idx = np.triu_indices_from(all_networks[0], 1)
    norm_networks = [np.arctanh(mat) for mat in all_networks]
    
    for mat in norm_networks:
        mat[np.isinf(mat)] = 0
        mat[np.isnan(mat)] = 0
        
    vec_networks = [mat[idx] for mat in norm_networks]
    return np.vstack(vec_networks)

# --- Utilitaires ---
def feature_selection(matrix, labels, train_ind, fnum):
    estimator = RidgeClassifier()
    selector = RFE(estimator, n_features_to_select=fnum, step=100, verbose=1)
    featureX = matrix[train_ind, :]
    featureY = labels[train_ind]
    selector = selector.fit(featureX, featureY.ravel())
    x_data = selector.transform(matrix)
    print("Features selected:", x_data.shape[1])
    return x_data

def site_percentage(train_ind, perc, subject_list):
    train_list = subject_list[train_ind]
    sites = get_subject_score(train_list, score='SITE_ID')
    unique = np.unique(list(sites.values())).tolist()
    site = np.array([unique.index(sites[train_list[x]]) for x in range(len(train_list))])
    labeled_indices = []
    for i in np.unique(site):
        id_in_site = np.argwhere(site == i).flatten()
        num_nodes = len(id_in_site)
        labeled_num = int(round(perc * num_nodes))
        labeled_indices.extend(train_ind[id_in_site[:labeled_num]])
    return labeled_indices

def create_affinity_graph_from_scores(scores, subject_list):
    num_nodes = len(subject_list)
    graph = np.zeros((num_nodes, num_nodes))
    for l in scores:
        label_dict = get_subject_score(subject_list, l)
        if l in ['AGE_AT_SCAN', 'FIQ']:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    try:
                        val = abs(float(label_dict[subject_list[k]]) - float(label_dict[subject_list[j]]))
                        if val < 2:
                            graph[k, j] += 1
                            graph[j, k] += 1
                    except (ValueError, KeyError): pass
        else:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    if subject_list[k] in label_dict and subject_list[j] in label_dict:
                        if label_dict[subject_list[k]] == label_dict[subject_list[j]]:
                            graph[k, j] += 1
                            graph[j, k] += 1
    return graph