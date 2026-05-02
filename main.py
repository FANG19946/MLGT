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
