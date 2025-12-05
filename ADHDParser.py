import os
import csv
import numpy as np
import scipy.io as sio
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import RFE

# Chemins
root_folder = os.getcwd()
data_folder = os.path.join(root_folder, 'data', 'ADHD')
phenotype_file = os.path.join(data_folder, 'phenotypic.csv')

def get_ids(num_subjects=None):
    subject_IDs = np.genfromtxt(os.path.join(data_folder, 'subject_IDs.txt'), dtype=str)
    if num_subjects is not None:
        subject_IDs = subject_IDs[:num_subjects]
    return subject_IDs

def get_subject_score(subject_list, score):
    # Mapping précis basé sur TON fichier csv
    key_map = {
        'DX_GROUP': 'adhd',    # Colonne 'adhd' (0 ou 1)
        'SITE_ID': 'site',     # Colonne 'site'
        'SEX': 'sex',          # Colonne 'sex'
        'AGE_AT_SCAN': 'age'   # Colonne 'age'
    }
    
    col_name = key_map.get(score, score)
    scores_dict = {}
    
    with open(phenotype_file) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            # L'ID est dans la colonne 'Subject'
            if row['Subject'] in subject_list:
                val = row[col_name]
                
                # Conversion Spécifique Diagnostic pour matcher ABIDE
                if score == 'DX_GROUP':
                    val = int(val)
                    # ABIDE utilise : 1=Autisme, 2=Control
                    # ADHD utilise : 1=ADHD, 0=Control
                    if val == 0:
                        val = 2 # Control devient 2
                    else:
                        val = 1 # ADHD (1, 2, 3) devient 1
                        
                scores_dict[row['Subject']] = val
                
    return scores_dict

def get_networks(subject_list, kind, atlas_name="ho", variable='connectivity'):
    all_networks = []
    for subject in subject_list:
        fl = os.path.join(data_folder, subject,
                          f"{subject}_{atlas_name}_{kind}.mat")
        try:
            matrix = sio.loadmat(fl)[variable]
            all_networks.append(matrix)
        except FileNotFoundError:
            print(f"Manquant: {fl}")
            
    # Vectorisation (Même logique que ABIDE)
    idx = np.triu_indices_from(all_networks[0], 1)
    norm_networks = [np.arctanh(mat) for mat in all_networks]
    # Sécurité contre les divisions par zéro / infinis
    for mat in norm_networks:
        mat[np.isinf(mat)] = 0
        mat[np.isnan(mat)] = 0
        
    vec_networks = [mat[idx] for mat in norm_networks]
    return np.vstack(vec_networks)

# --- Fonctions utilitaires (Identiques à ABIDEParser) ---

def feature_selection(matrix, labels, train_ind, fnum):
    estimator = RidgeClassifier()
    selector = RFE(estimator, n_features_to_select=fnum, step=100, verbose=1)
    featureX = matrix[train_ind, :]
    featureY = labels[train_ind]
    selector = selector.fit(featureX, featureY.ravel())
    return selector.transform(matrix)

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
                    except ValueError: pass
        else:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    # Comparaison de strings (ex: 'M' == 'M' ou 'Peking_1' == 'Peking_1')
                    if label_dict[subject_list[k]] == label_dict[subject_list[j]]:
                        graph[k, j] += 1
                        graph[j, k] += 1
    return graph