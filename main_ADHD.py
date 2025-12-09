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
    """
        train_ind       : indices of the training samples
        test_ind        : indices of the test samples
        val_ind         : indices of the validation samples
        graph_feat      : population graph computed from phenotypic measures
        features        : feature vectors
        y               : ground truth labels
        y_data          : ground truth labels (one-hot)
        params          : dictionnary of GCNs parameters
        subject_IDs     : list of subject IDs
    """

    print(len(train_ind))

    # selection of a subset of data if running experiments with a subset of the training set
    labeled_ind = Reader.site_percentage(train_ind, params['num_training'], subject_IDs)

    # feature selection/dimensionality reduction step
    x_data = Reader.feature_selection(features, y, labeled_ind, params['num_features'])

    fold_size = len(test_ind)

    # Calculate all pairwise distances (Visual Similarity Graph)
    distv = distance.pdist(x_data, metric='correlation')
    # Convert to a square symmetric distance matrix
    dist = distance.squareform(distv)
    sigma = np.mean(dist)
    # Get affinity from similarity matrix
    sparse_graph = np.exp(- dist ** 2 / (2 * sigma ** 2))
    
    # Combine with phenotypic graph
    final_graph = graph_feat * sparse_graph

    # Linear classifier (Baseline)
    clf = RidgeClassifier()
    clf.fit(x_data[train_ind, :], y[train_ind].ravel())
    
    # Compute the accuracy
    lin_acc = clf.score(x_data[test_ind, :], y[test_ind].ravel())
    
    # Compute the AUC
    pred = clf.decision_function(x_data[test_ind, :])
    try:
        lin_auc = sklearn.metrics.roc_auc_score(y[test_ind] - 1, pred)
    except:
        lin_auc = 0.5 
        
    print("Linear Accuracy: " + str(lin_acc))

    # Classification with GCNs
    test_acc, test_auc = Train.run_training(final_graph, sparse.coo_matrix(x_data).tolil(), y_data, train_ind, val_ind,
                                            test_ind, params)

    print(test_acc)

    # return number of correctly classified samples instead of percentage
    test_acc = int(round(test_acc * len(test_ind)))
    lin_acc = int(round(lin_acc * len(test_ind)))

    return test_acc, test_auc, lin_acc, lin_auc, fold_size


def main():
    parser = argparse.ArgumentParser(description='Graph CNNs for population graphs: classification of the ADHD dataset')
    parser.add_argument('--dropout', default=0.5, type=float, help='Dropout rate (1 - keep probability) (default: 0.3)')
    parser.add_argument('--decay', default=5e-3, type=float, help='Weight for L2 loss on embedding matrix (default: 5e-4)')
    parser.add_argument('--hidden', default=8, type=int, help='Number of filters in hidden layers (default: 16)')
    parser.add_argument('--lrate', default=0.005, type=float, help='Initial learning rate (default: 0.005)')
    parser.add_argument('--atlas', default='aal', help='atlas for network construction (node definition) (default: aal)')
    parser.add_argument('--suffix', type=str, default="", help='Suffixe ajouté au nom du fichier de résultats (.mat)')
    parser.add_argument('--epochs', default=150, type=int, help='Number of epochs to train')
    parser.add_argument('--num_features', default=200, type=int, help='Number of features to keep for the feature selection step')
    parser.add_argument('--num_training', default=1.0, type=float, help='Percentage of training set used for training')
    parser.add_argument('--depth', default=0, type=int, help='Number of additional hidden layers in the GCN.')
    parser.add_argument('--model', default='gcn_cheby', help='gcn model used (default: gcn_cheby)')
    parser.add_argument('--seed', default=123, type=int, help='Seed for random initialisation (default: 123)')
    parser.add_argument('--folds', default=11, type=int, help='For cross validation, specifies which fold will be used. All folds are used if set to 11')
    parser.add_argument('--save', default=1, type=int, help='Parameter that specifies if results have to be saved.')
    parser.add_argument('--connectivity', default='correlation', help='Type of connectivity used for network construction')

    args = parser.parse_args()
    
    # GCN Parameters
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
    
    # Filtrage spécifique ADHD : On garde seulement ceux qui ont un label
    # (Important car ADHDParser peut retourner des IDs qui n'ont pas de label associé dans le CSV)
    subject_IDs = np.array([s for s in subject_IDs if s in labels])
    
    num_nodes = len(subject_IDs)
    num_classes = 2
    
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
        
    # Graphe phénotypique : Sexe et Site
    #graph = Reader.create_affinity_graph_from_scores(['SEX', 'SITE_ID'], subject_IDs)
    graph = Reader.create_affinity_graph_from_scores(['SEX', 'SITE_ID', 'AGE_AT_SCAN'], subject_IDs, sigma=100000000) #pour age gauss

    # Cross Validation
    skf = StratifiedKFold(n_splits=10)

    if args.folds == 11:  # run cross validation on all folds
        scores = Parallel(n_jobs=10)(delayed(train_fold)(train_ind, test_ind, test_ind, graph, features, y, y_data,
                                                         params, subject_IDs)
                                     for train_ind, test_ind in
                                     reversed(list(skf.split(np.zeros(num_nodes), np.squeeze(y)))))

        print(scores)

        scores_acc = [x[0] for x in scores]
        scores_auc = [x[1] for x in scores]
        scores_lin = [x[2] for x in scores]
        scores_auc_lin = [x[3] for x in scores]
        fold_size = [x[4] for x in scores]

        # Prints EXACTEMENT comme ABIDE
        print('overall linear accuracy %f' + str(np.sum(scores_lin) * 1. / num_nodes))
        print('overall linear AUC %f' + str(np.mean(scores_auc_lin)))
        print('overall accuracy %f' + str(np.sum(scores_acc) * 1. / num_nodes))
        print('overall AUC %f' + str(np.mean(scores_auc)))

    else:  # compute results for only one fold
        cv_splits = list(skf.split(features, np.squeeze(y)))
        train_ind = cv_splits[args.folds][0]
        test_ind = cv_splits[args.folds][1]
        val_ind = test_ind # On utilise le test comme validation ici (comme ABIDE)

        scores_acc, scores_auc, scores_lin, scores_auc_lin, fold_size = train_fold(
            train_ind, test_ind, val_ind, graph, features, y, y_data, params, subject_IDs
        )
        
        # Prints EXACTEMENT comme ABIDE pour un fold unique
        print('overall linear accuracy %f' + str(np.sum(scores_lin) * 1. / fold_size))
        print('overall linear AUC %f' + str(np.mean(scores_auc_lin)))
        print('overall accuracy %f' + str(np.sum(scores_acc) * 1. / fold_size))
        print('overall AUC %f' + str(np.mean(scores_auc)))

    # Sauvegarde des résultats
    root = Path(os.getcwd()).resolve()
    if args.save == 1:
        save_dir = root / "results"
        save_dir.mkdir(parents=True, exist_ok=True)

        result_name = 'ADHD_classification' 
        if args.suffix != "":
            result_name = f"{result_name}_{args.suffix}"
        
        out_path = save_dir / f"{result_name}.mat"
        print("Saving results to:", out_path)

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