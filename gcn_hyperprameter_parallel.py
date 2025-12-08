import os
import csv
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.spatial import distance
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import RidgeClassifier
import sklearn.metrics
from joblib import Parallel, delayed  # <- parallelisation

import ABIDEParser as Reader
import train_GCN as Train


# ---------- Data building ----------

def build_abide_data(connectivity="correlation", atlas="ho"):
    """
    Load ABIDE data, build labels, features, and phenotypic graph.
    """
    # Subject IDs
    subject_IDs = Reader.get_ids()

    # Clinical labels: DX_GROUP (1 = control, 2 = ASD)
    labels = Reader.get_subject_score(subject_IDs, score='DX_GROUP')

    # Acquisition sites
    sites = Reader.get_subject_score(subject_IDs, score='SITE_ID')
    unique_sites = np.unique(list(sites.values())).tolist()

    num_classes = 2
    num_nodes = len(subject_IDs)

    # One-hot labels (y_data) and scalar labels (y) and site index
    y_data = np.zeros([num_nodes, num_classes], dtype=float)
    y = np.zeros([num_nodes, 1], dtype=int)
    site = np.zeros([num_nodes, 1], dtype=int)

    for i in range(num_nodes):
        lab = int(labels[subject_IDs[i]])
        y_data[i, lab - 1] = 1
        y[i] = lab
        site[i] = unique_sites.index(sites[subject_IDs[i]])

    # Connectivity features (vectorised networks)
    features = Reader.get_networks(subject_IDs, kind=connectivity, atlas_name=atlas)

    # Phenotypic population graph (e.g. SEX + SITE_ID)
    phenotypic_graph = Reader.create_affinity_graph_from_scores(['SEX', 'SITE_ID'], subject_IDs)

    return subject_IDs, features, y, y_data, phenotypic_graph


def build_final_graph_and_features(features, y, train_ind, params, subject_IDs, phenotypic_graph):
    """
    Per-split pipeline from the original code:
    - site_percentage (subset of training)
    - feature_selection
    - similarity graph from imaging features
    - final_graph = phenotypic_graph * similarity_graph
    """
    # Optionally use only a percentage of the training set for feature selection
    labeled_ind = Reader.site_percentage(train_ind, params['num_training'], subject_IDs)

    # Feature selection / dimensionality reduction (e.g. to 2000 features)
    x_data = Reader.feature_selection(features, y, labeled_ind, params['num_features'])

    # Pairwise distances in feature space
    distv = distance.pdist(x_data, metric='correlation')
    dist = distance.squareform(distv)
    sigma = np.mean(dist)

    # Gaussian affinity from feature distances
    sparse_graph = np.exp(- dist ** 2 / (2 * sigma ** 2))

    # Combine phenotypic graph and feature similarity graph
    final_graph = phenotypic_graph * sparse_graph

    return final_graph, x_data


# ---------- Single-fold training ----------

def train_single_split(train_ind, test_ind,
                       features, y, y_data, subject_IDs,
                       phenotypic_graph, params):
    """
    Single train/test split:
    - build final_graph + x_data
    - linear baseline
    - GCN / ChebNet via Train.run_training
    """
    # Build final graph + features for this split
    final_graph, x_data = build_final_graph_and_features(
        features, y, train_ind, params, subject_IDs, phenotypic_graph
    )

    # ----- Linear baseline -----
    clf = RidgeClassifier()
    clf.fit(x_data[train_ind, :], y[train_ind].ravel())
    lin_acc = clf.score(x_data[test_ind, :], y[test_ind].ravel())
    pred = clf.decision_function(x_data[test_ind, :])
    lin_auc = sklearn.metrics.roc_auc_score(y[test_ind] - 1, pred)

    # ----- GCN / ChebNet -----
    # Following original script: use test as validation as well
    val_ind = test_ind

    test_acc, test_auc = Train.run_training(
        final_graph,
        sparse.coo_matrix(x_data).tolil(),
        y_data,
        train_ind, val_ind, test_ind,
        params
    )

    return float(test_acc), float(test_auc), float(lin_acc), float(lin_auc), len(test_ind)


# Wrapper to be used by joblib (one fold)
def run_fold_for_config(fold_id, train_idx, test_idx,
                        features, y, y_data, subject_IDs,
                        phenotypic_graph, params,
                        model, hidden, depth, K):
    """
    Run one fold for a given config and return a dict (one CSV row).
    """
    start = time.time()
    test_acc, test_auc, lin_acc, lin_auc, fold_size = train_single_split(
        train_idx, test_idx,
        features, y, y_data,
        subject_IDs,
        phenotypic_graph,
        params
    )
    elapsed = time.time() - start

    print(f"  Fold {fold_id}/10: "
          f"GCN ACC={test_acc:.4f}, AUC={test_auc:.4f} | "
          f"LIN ACC={lin_acc:.4f}, AUC={lin_auc:.4f} | "
          f"Time={elapsed:.2f}s")

    return {
        "model": model,
        "hidden": hidden,
        "depth": depth,
        "cheby_K": K if model == 'gcn_cheby' else None,
        "cv": fold_id,
        "test_acc": test_acc,
        "test_auc": test_auc,
        "lin_acc": lin_acc,
        "lin_auc": lin_auc,
        "n_test": fold_size,
        "time_sec": elapsed,
    }


def get_base_params():
    """
    Base hyperparameters (kept fixed across all architecture experiments),
    matching the ABIDE configuration from the paper.
    """
    params = dict()
    params['lrate'] = 0.005          # learning rate
    params['epochs'] = 150          # max epochs
    params['dropout'] = 0.3          # dropout rate
    params['hidden'] = 16            # will be overridden in the grid
    params['decay'] = 5e-4           # L2 weight decay
    params['early_stopping'] = 150   # effectively disables early stopping
    params['max_degree'] = 5         # Chebyshev K; overridden for cheby models
    params['depth'] = 0              # additional hidden layers; overridden in the grid
    params['seed'] = 123             # random seed
    params['num_features'] = 2000    # feature selection size
    params['num_training'] = 1.0     # use full training set
    params['model'] = 'gcn_cheby'    # overridden in the grid
    return params


def main():
    # ---------- 1. Load ABIDE data ----------
    subject_IDs, features, y, y_data, phenotypic_graph = build_abide_data(
        connectivity="correlation",
        atlas="ho"
    )
    num_nodes = len(subject_IDs)
    all_indices = np.arange(num_nodes)
    y_flat = np.squeeze(y)

    # ---------- 2. 10-fold stratified cross-validation ----------
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
    # Precompute splits so they are reused identically for all configs
    fold_splits = list(skf.split(all_indices, y_flat))

    # ---------- 3. Define architecture hyperparameter grid ----------
    model_list = ['gcn', 'gcn_cheby', 'gcnii']   # Kipf vs Chebyshev vs GCNII
    hidden_list = [16, 32, 64]                  # hidden dimension
    depth_list = [0, 1, 2]                      # total hidden layers = 1 + depth
    cheby_K_list = [1, 3, 5]                    # relevant only for gcn_cheby

    # How many folds to run in parallel for each config
    N_JOBS = 10  # you can set to e.g. os.cpu_count()

    all_rows = []  # one row per (config, fold)

    # ---------- 4. Run experiments with 10-fold CV (parallel over folds) ----------
    for model in model_list:
        for hidden in hidden_list:
            for depth in depth_list:
                if model == 'gcn_cheby':
                    K_values = cheby_K_list
                else:
                    K_values = [1]  # dummy K for GCN / GCNII

                for K in K_values:
                    params = get_base_params()
                    params['model'] = model
                    params['hidden'] = hidden
                    params['depth'] = depth
                    params['max_degree'] = K

                    print("\n============================================")
                    print(f"Config: model={model}, hidden={hidden}, depth={depth}, K={K}")
                    print("============================================")

                    config_start = time.time()

                    # Run all folds for this config in parallel
                    fold_rows = Parallel(n_jobs=N_JOBS)(
                        delayed(run_fold_for_config)(
                            fold_id=fold_id + 1,
                            train_idx=train_idx,
                            test_idx=test_idx,
                            features=features,
                            y=y,
                            y_data=y_data,
                            subject_IDs=subject_IDs,
                            phenotypic_graph=phenotypic_graph,
                            params=params,
                            model=model,
                            hidden=hidden,
                            depth=depth,
                            K=K
                        )
                        for fold_id, (train_idx, test_idx) in enumerate(fold_splits)
                    )

                    config_elapsed = time.time() - config_start
                    print(f"Total time for config -> {config_elapsed:.2f}s")

                    all_rows.extend(fold_rows)

    # ---------- 5. Save per-fold CV results to CSV ----------
    root = Path(os.getcwd()).resolve()
    save_dir = root / "results"
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "gcn_gcnii_10cv.csv"

    fieldnames = [
        "model", "hidden", "depth", "cheby_K",
        "cv",
        "test_acc", "test_auc",
        "lin_acc", "lin_auc",
        "n_test", "time_sec"
    ]

    with open(out_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print("\nAll experiments finished.")
    print("Per-fold CV results saved to:", out_path)


if __name__ == "__main__":
    main()
