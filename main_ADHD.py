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
    
    # Construction du graphe de similarité visuelle
    distv = distance.pdist(x_data, metric='correlation')
    dist = distance.squareform(distv)
    sigma = np.mean(dist)
    sparse_graph = np.exp(- dist ** 2 / (2 * sigma ** 2))
    
    # Fusion avec le graphe phénotypique
    final_graph = graph_feat * sparse_graph
    
    # --- ABLATION STUDY (Décommenter pour tester sans graphe) ---
    # final_graph = np.eye(len(final_graph))
    # -----------------------------------------------------------

    # Classifieur Linéaire (Baseline)
    clf = RidgeClassifier()
    clf.fit(x_data[train_ind, :], y[train_ind].ravel())
    lin_acc = clf.score(x_data[test_ind, :], y[test_ind].ravel())
    pred = clf.decision_function(x_data[test_ind, :])
    try:
        lin_auc = sklearn.metrics.roc_auc_score(y[test_ind] - 1, pred)
    except:
        lin_auc = 0.5 
        
    print("Linear Accuracy: " + str(lin_acc))

    # GCN
    test_acc, test_auc = Train.run_training(final_graph, sparse.coo_matrix(x_data).tolil(), y_data, train_ind, val_ind,
                                            test_ind, params)

    test_acc = int(round(test_acc * len(test_ind)))
    lin_acc = int(round(lin_acc * len(test_ind)))

    return test_acc, test_auc, lin_acc, lin_auc, len(test_ind)

def main():

    parser = argparse.ArgumentParser(description='Graph CNNs for population graphs: '
                                                 'classification of the ABIDE dataset')
    parser.add_argument('--dropout', default=0.3, type=float,
                        help='Dropout rate (1 - keep probability) (default: 0.3)')
    parser.add_argument('--decay', default=5e-4, type=float,
                        help='Weight for L2 loss on embedding matrix (default: 5e-4)')
    parser.add_argument('--hidden', default=16, type=int, help='Number of filters in hidden layers (default: 16)')
    parser.add_argument('--lrate', default=0.005, type=float, help='Initial learning rate (default: 0.005)')
    parser.add_argument('--atlas', default='ho', help='atlas for network construction (node definition) (default: ho, '
                                                      'see preprocessed-connectomes-project.org/abide/Pipelines.html '
                                                      'for more options )')
    #parser.add_argument('--atlas', default='aal', help='Atlas name (default: aal)')
    parser.add_argument('--suffix', type=str, default="",
                    help='Suffixe ajouté au nom du fichier de résultats (.mat)')
    parser.add_argument('--epochs', default=150, type=int, help='Number of epochs to train')
    parser.add_argument('--num_features', default=2000, type=int, help='Number of features to keep for '
                                                                       'the feature selection step (default: 2000)')
    parser.add_argument('--num_training', default=1.0, type=float, help='Percentage of training set used for '
                                                                        'training (default: 1.0)')
    parser.add_argument('--depth', default=0, type=int, help='Number of additional hidden layers in the GCN. '
                                                             'Total number of hidden layers: 1+depth (default: 0)')
    parser.add_argument('--model', default='gcn_cheby', help='gcn model used (default: gcn_cheby, '
                                                             'uses chebyshev polynomials, '
                                                             'options: gcn, gcn_cheby, dense )')
    parser.add_argument('--seed', default=123, type=int, help='Seed for random initialisation (default: 123)')
    parser.add_argument('--folds', default=0, type=int, help='For cross validation, specifies which fold will be '
                                                             'used. All folds are used if set to 11 (default: 11)')
    parser.add_argument('--save', default=1, type=int, help='Parameter that specifies if results have to be saved. '
                                                            'Results will be saved if set to 1 (default: 1)')
    parser.add_argument('--connectivity', default='correlation', help='Type of connectivity used for network '
                                                                      'construction (default: correlation, '
                                                                      'options: correlation, partial correlation, '
                                                                      'tangent)')

    args = parser.parse_args()
    
    # GCN Parameters
    params = dict()
    params['model'] = args.model                    # gcn model using chebyshev polynomials
    params['lrate'] = args.lrate                    # Initial learning rate
    params['epochs'] = args.epochs                  # Number of epochs to train
    params['dropout'] = args.dropout                # Dropout rate (1 - keep probability)
    params['hidden'] = args.hidden                  # Number of units in hidden layers
    params['decay'] = args.decay                    # Weight for L2 loss on embedding matrix.
    params['early_stopping'] = params['epochs']     # Tolerance for early stopping (# of epochs). No early stopping if set to param.epochs
    #params['early_stopping'] = 15
    params['max_degree'] = 3                        # Maximum Chebyshev polynomial degree.
    params['depth'] = args.depth                    # number of additional hidden layers in the GCN. Total number of hidden layers: 1+depth
    params['seed'] = args.seed                      # seed for random initialisation
    params['num_features'] = args.num_features
    params['num_training'] = args.num_training
    atlas = args.atlas
    connectivity = args.connectivity

    

    # Chargement des données
    subject_IDs = Reader.get_ids()
    labels = Reader.get_subject_score(subject_IDs, score='DX_GROUP')
    sites = Reader.get_subject_score(subject_IDs, score='SITE_ID')
    
    num_classes = 2
    num_nodes = len(subject_IDs)
    
    # Filtrage des sujets qui n'ont pas de label (si le CSV est incomplet)
    # On garde seulement ceux qui sont dans 'labels'
    subject_IDs = np.array([s for s in subject_IDs if s in labels])
    num_nodes = len(subject_IDs)
    print(f"Nombre de sujets valides (avec Matrice + Label) : {num_nodes}")

    y_data = np.zeros([num_nodes, num_classes])
    y = np.zeros([num_nodes, 1])
    site = np.zeros([num_nodes, 1], dtype=int)
    
    # Gestion des sites (conversion string -> int index)
    unique_sites = sorted(list(set([sites[s] for s in subject_IDs if s in sites])))

    # Matrice de features
    features = Reader.get_networks(subject_IDs, kind=connectivity, atlas_name=atlas)

    # Préparation des labels et sites
    for i in range(num_nodes):
        sid = subject_IDs[i]
        lbl = int(labels[sid])
        y_data[i, lbl-1] = 1
        y[i] = lbl
        if sid in sites:
            site[i] = unique_sites.index(sites[sid])
        
    # Graphe phénotypique : Sexe et Site (Hypothèse 1 - Baseline)
    #graph = Reader.create_affinity_graph_from_scores(['SEX'], subject_IDs)
    #graph = Reader.create_affinity_graph_from_scores(['SITE_ID'], subject_IDs)
    graph = Reader.create_affinity_graph_from_scores(['SEX', 'SITE_ID'], subject_IDs) #baseline parisot
    #graph = Reader.create_affinity_graph_from_scores(['SITE_ID', 'SEX', 'AGE_AT_SCAN'], subject_IDs)
    #graph = Reader.create_affinity_graph_from_scores(['SITE_ID', 'SEX', 'FIQ'], subject_IDs)
    #graph = Reader.create_affinity_graph_from_scores(['SITE_ID', 'SEX', 'MED_STATUS'], subject_IDs)
    #graph = Reader.create_affinity_graph_from_scores(['SITE_ID', 'SEX', 'AGE_AT_SCAN', 'MED_STATUS'], subject_IDs)
    #graph = Reader.create_affinity_graph_from_scores(['SITE_ID', 'SEX', 'HANDEDNESS'], subject_IDs)


    # Cross Validation
    skf = StratifiedKFold(n_splits=10)

    if args.folds == 11:
        print("Running 10-fold CV...")
        scores = Parallel(n_jobs=10)(delayed(train_fold)(train_ind, test_ind, test_ind, graph, features, y, y_data,
                                                         params, subject_IDs)
                                     for train_ind, test_ind in
                                     reversed(list(skf.split(np.zeros(num_nodes), np.squeeze(y)))))
        
        # ON CRÉE LES VARIABLES COMME DANS ABIDE
        scores_acc = [x[0] for x in scores]
        scores_auc = [x[1] for x in scores]
        scores_lin = [x[2] for x in scores]      # <--- C'était manquant
        scores_auc_lin = [x[3] for x in scores]  # <--- C'était manquant
        fold_size = [x[4] for x in scores]       # <--- C'était manquant

        print(f'Overall Accuracy: {np.sum(scores_acc) * 1. / num_nodes}')
        print(f'Overall AUC: {np.mean(scores_auc)}')

    else:
        print(f"Running single fold {args.folds}...")
        cv_splits = list(skf.split(features, np.squeeze(y)))
        train_ind = cv_splits[args.folds][0]
        test_ind = cv_splits[args.folds][1]
        
        # ON CRÉE LES VARIABLES COMME DANS ABIDE
        scores_acc, scores_auc, scores_lin, scores_auc_lin, fold_size = train_fold(
            train_ind, test_ind, test_ind, graph, features, y, y_data, params, subject_IDs
        )
        
        print(f"Fold Accuracy: {scores_acc / fold_size}") # Note: scores_acc est un entier ici

    # --- SAUVEGARDE (EXACTEMENT COMME ABIDE) ---
    root = Path(os.getcwd()).resolve()
    if args.save == 1:
        save_dir = root / "results"
        save_dir.mkdir(parents=True, exist_ok=True)

        result_name = 'ADHD_classification' 
        if args.suffix != "":
            result_name = f"{result_name}_{args.suffix}"
        
        out_path = save_dir / f"{result_name}.mat"
        print("Saving results to:", out_path)

        # Maintenant vous pouvez utiliser la syntaxe simple car les variables existent
        sio.savemat(
            out_path,
            {
                'lin': scores_lin,
                'lin_auc': scores_auc_lin,
                'acc': scores_acc,
                'auc': scores_auc,
                'folds': fold_size
            }
        )

if __name__ == "__main__":
    main()