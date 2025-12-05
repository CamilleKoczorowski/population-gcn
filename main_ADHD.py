import time
import argparse
import numpy as np
from scipy import sparse
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from scipy.spatial import distance
from sklearn.linear_model import RidgeClassifier
import sklearn.metrics
import scipy.io as sio
import os
from pathlib import Path

# Imports locaux
import train_GCN as Train
import ADHDParser as Reader 

def train_fold(train_ind, test_ind, val_ind, graph_feat, features, y, y_data, params, subject_IDs):
    print(f"Fold training size: {len(train_ind)}")
    
    labeled_ind = Reader.site_percentage(train_ind, params['num_training'], subject_IDs)
    x_data = Reader.feature_selection(features, y, labeled_ind, params['num_features'])
    
    fold_size = len(test_ind)
    
    # Construction du graphe de similarité visuelle
    distv = distance.pdist(x_data, metric='correlation')
    dist = distance.squareform(distv)
    sigma = np.mean(dist)
    sparse_graph = np.exp(- dist ** 2 / (2 * sigma ** 2))
    
    # Fusion avec le graphe phénotypique
    final_graph = graph_feat * sparse_graph
    
    # --- ABLATION STUDY (Décommenter pour l'étape 3) ---
    # final_graph = np.eye(len(final_graph))
    # ---------------------------------------------------

    # Classifieur Linéaire (Baseline)
    clf = RidgeClassifier()
    clf.fit(x_data[train_ind, :], y[train_ind].ravel())
    lin_acc = clf.score(x_data[test_ind, :], y[test_ind].ravel())
    pred = clf.decision_function(x_data[test_ind, :])
    try:
        lin_auc = sklearn.metrics.roc_auc_score(y[test_ind] - 1, pred)
    except:
        lin_auc = 0.5 # Cas où une seule classe est présente
        
    print("Linear Accuracy: " + str(lin_acc))

    # GCN
    test_acc, test_auc = Train.run_training(final_graph, sparse.coo_matrix(x_data).tolil(), y_data, train_ind, val_ind,
                                            test_ind, params)

    test_acc = int(round(test_acc * len(test_ind)))
    lin_acc = int(round(lin_acc * len(test_ind)))

    return test_acc, test_auc, lin_acc, lin_auc, fold_size

def main():
    parser = argparse.ArgumentParser(description='Graph CNNs for ADHD dataset')
    parser.add_argument('--dropout', default=0.3, type=float)
    parser.add_argument('--decay', default=5e-4, type=float)
    parser.add_argument('--hidden', default=16, type=int)
    parser.add_argument('--lrate', default=0.005, type=float)
    parser.add_argument('--atlas', default='ho') 
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--num_features', default=2000, type=int)
    parser.add_argument('--num_training', default=1.0, type=float)
    parser.add_argument('--depth', default=0, type=int)
    parser.add_argument('--model', default='gcn_cheby')
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--folds', default=11, type=int)
    parser.add_argument('--connectivity', default='correlation') 

    args = parser.parse_args()
    
    params = dict()
    params['model'] = args.model
    params['lrate'] = args.lrate
    params['epochs'] = args.epochs
    params['dropout'] = args.dropout
    params['hidden'] = args.hidden
    params['decay'] = args.decay
    params['early_stopping'] = params['epochs']
    params['max_degree'] = 3
    params['depth'] = args.depth
    params['seed'] = args.seed
    params['num_features'] = args.num_features
    params['num_training'] = args.num_training
    atlas = args.atlas
    connectivity = args.connectivity

    # Chargement des données
    subject_IDs = Reader.get_ids()
    labels = Reader.get_subject_score(subject_IDs, score='DX_GROUP')
    sites = Reader.get_subject_score(subject_IDs, score='SITE_ID')
    
    # Gestion de cas où il y aurait des sites inconnus ou NaN
    unique_sites = sorted(list(set(sites.values())))
    
    num_classes = 2
    num_nodes = len(subject_IDs)

    y_data = np.zeros([num_nodes, num_classes])
    y = np.zeros([num_nodes, 1])
    
    # Matrice de features
    features = Reader.get_networks(subject_IDs, kind=connectivity, atlas_name=atlas)

    # Préparation des labels et sites
    for i in range(num_nodes):
        # label: 1 (ADHD) ou 2 (Control) -> index 0 ou 1
        lbl = int(labels[subject_IDs[i]])
        y_data[i, lbl-1] = 1
        y[i] = lbl
        
    # Graphe phénotypique (Sexe et Site)
    graph = Reader.create_affinity_graph_from_scores(['SEX', 'SITE_ID'], subject_IDs)

    # Cross Validation
    skf = StratifiedKFold(n_splits=10)

    if args.folds == 11:
        print("Running 10-fold CV...")
        scores = Parallel(n_jobs=10)(delayed(train_fold)(train_ind, test_ind, test_ind, graph, features, y, y_data,
                                                         params, subject_IDs)
                                     for train_ind, test_ind in
                                     reversed(list(skf.split(np.zeros(num_nodes), np.squeeze(y)))))
        
        scores_acc = [x[0] for x in scores]
        scores_auc = [x[1] for x in scores]
        print(f'Overall Accuracy: {np.sum(scores_acc) * 1. / num_nodes}')
        print(f'Overall AUC: {np.mean(scores_auc)}')
    else:
        print(f"Running single fold {args.folds}...")
        cv_splits = list(skf.split(features, np.squeeze(y)))
        train_ind = cv_splits[args.folds][0]
        test_ind = cv_splits[args.folds][1]
        
        train_fold(train_ind, test_ind, test_ind, graph, features, y, y_data, params, subject_IDs)

if __name__ == "__main__":
    main()
