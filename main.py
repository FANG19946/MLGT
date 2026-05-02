import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm
import torch

from utils import (
    build_testing_matrix,
    dataset_params,
    train_classifiers,
    evaluation_metrics
)

def load_extreme_data(file_path):
    with open(file_path, 'r') as f:
        header = f.readline().split()
        n_samples, n_features, n_labels = map(int, header)

        labels = []
        feat_rows, feat_cols, feat_vals = [], [], []

        for i, line in enumerate(f):
            parts = line.strip().split()
            if not parts:
                continue

            if ':' not in parts[0]:
                for label in parts[0].split(','):
                    lab = int(label)
                    if lab >= n_labels:
                        lab -= 1
                    labels.append((i, lab)) 
                feature_start = 1
            else:
                feature_start = 0

            for feat in parts[feature_start:]:
                idx, val = feat.split(':')
                feat_rows.append(i)
                col = int(idx)
                if col >= n_features:
                    col -= 1  
                feat_cols.append(col)
                feat_vals.append(float(val))

    X = csr_matrix((feat_vals, (feat_rows, feat_cols)),
                   shape=(n_samples, n_features))

    if labels:
        rows, cols = zip(*labels)
        Y = csr_matrix((np.ones(len(rows)), (rows, cols)),
                       shape=(n_samples, n_labels))
    else:
        Y = csr_matrix((n_samples, n_labels))

    return X, Y

def run_experiment(dataset_name='mediamill'):

    X, Y = load_extreme_data('Mediamill_data.txt')

    train_idx = np.loadtxt('mediamill_trSplit.txt', dtype=int).flatten() - 1
    test_idx = np.loadtxt('mediamill_tstSplit.txt', dtype=int).flatten() - 1

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]

    e_ratio = 0.5
    n_labels, _, k, e = dataset_params(Y_train, e_ratio)

    C1 = 8
    C2 = 2

    e = int(C2 * k * np.log(n_labels))

    n_tests = int(10 * k * np.log(n_labels + 1))

    print(f"\nDataset stats:")
    print(f"n_labels = {n_labels}, n_tests = {n_tests}, k = {k}, e = {e}")

    methods = ['identity', 'bernoulli', 'expander', 'rs' , 'nmf']

    results = {}

    for method in tqdm(methods, desc="Matrix methods"):

        print(f"\n--- {method} ---")

        A,n_tests,e = build_testing_matrix(
            
            n_labels=n_labels,
            k=k,
            method=method,
            seed=42 , 
            Y_train = Y_train
        )

        W = train_classifiers(
            dataset=(X_train, Y_train),
            A=A,
            epochs=30,
            lr=0.001,
            device='cuda'  
        )

        metrics = evaluation_metrics(
            W=W,
            dataset=(X_test, Y_test),
            A=A,
            k=k,
            e=e,
            device='cuda'
        )

        results[method] = metrics
        del W
        torch.cuda.empty_cache()

    print("\n===== FINAL RESULTS =====")
    for method, metrics in results.items():
        print(f"{method}: HL={metrics['hamming_loss']:.4f}, P@{k}={metrics['precision@k']:.4f}, n_labels={metrics['n_labels']}, n_tests={metrics['n_tests']}")


if __name__ == "__main__":
    run_experiment('mediamill')