import numpy as np
from skmultilearn.dataset import load_dataset
from sklearn.preprocessing import normalize
from tqdm import tqdm
import torch

from utils import (
    build_testing_matrix,
    dataset_params,
    train_classifiers,
    evaluation_metrics
)


# ----------------------------
# LOAD DATASET
# ----------------------------
import urllib.request
import os
from scipy.io import arff
import numpy as np


def download_if_needed(url, path):
    if not os.path.exists(path):
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, path)


def load_arff_dataset(train_url, test_url, name):

    train_path = f"{name}_train.arff"
    test_path = f"{name}_test.arff"

    download_if_needed(train_url, train_path)
    download_if_needed(test_url, test_path)

    train_data, _ = arff.loadarff(train_path)
    test_data, _ = arff.loadarff(test_path)

    train_data = np.array(train_data.tolist())
    test_data = np.array(test_data.tolist())

    return train_data, test_data

from skmultilearn.dataset import load_from_arff
import os

def load_data(dataset_name="mediamill"):

    path = os.path.join("datasets", dataset_name)

    print("DEBUG PATH:", path)

    train_path = os.path.join(path, f"{dataset_name}-train.arff")
    test_path = os.path.join(path, f"{dataset_name}-test.arff")

    print("TRAIN PATH:", train_path)
    print("TEST PATH:", test_path)

    if not os.path.exists(train_path):
        raise FileNotFoundError(train_path)

    if not os.path.exists(test_path):
        raise FileNotFoundError(test_path)

    from skmultilearn.dataset import load_from_arff

    X_train, Y_train = load_from_arff(
        train_path,
        label_count=101,
        label_location="end",
        load_sparse=False
    )

    X_test, Y_test = load_from_arff(
        test_path,
        label_count=101,
        label_location="end",
        load_sparse=False
    )

    return (X_train, Y_train), (X_test, Y_test)
# ----------------------------
# MAIN EXPERIMENT
# ----------------------------
def run_experiment(dataset_name='mediamill'):

    print(f"\nLoading dataset: {dataset_name}")
    train_data, test_data = load_data(dataset_name)

    X_train, Y_train = train_data
    X_test, Y_test = test_data

    # ----------------------------
    # params
    # ----------------------------
    e_ratio = 0.5
    n_labels, _, k, e = dataset_params(Y_train, e_ratio)

    C1 = 8
    C2 = 2

    n_tests = int(C1 * k * np.log(n_labels))
    e = int(C2 * k * np.log(n_labels))

    # MLGT-style number of tests
    n_tests = int(10 * k * np.log(n_labels + 1))

    print(f"\nDataset stats:")
    print(f"n_labels = {n_labels}, n_tests = {n_tests}, k = {k}, e = {e}")

    # ----------------------------
    # methods
    # ----------------------------
    methods = ['identity', 'bernoulli', 'expander', 'rs']

    results = {}

    # ----------------------------
    # loop
    # ----------------------------
    for method in tqdm(methods, desc="Matrix methods"):

        print(f"\n--- {method} ---")

        A,n_tests,e = build_testing_matrix(
            
            n_labels=n_labels,
            k=k,
            method=method,
            seed=42
        )

        W = train_classifiers(
            dataset=(X_train, Y_train),
            A=A,
            epochs=30,
            lr=0.001,
            device='cuda'  # change if needed
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

    # ----------------------------
    # summary
    # ----------------------------
    print("\n===== FINAL RESULTS =====")
    for method, metrics in results.items():
        print(f"{method}: HL={metrics['hamming_loss']:.4f}, P@{k}={metrics['precision@k']:.4f}, n_labels={metrics['n_labels']}, n_tests={metrics['n_tests']}")


# ----------------------------
# RUN
# ----------------------------
if __name__ == "__main__":
    run_experiment('mediamill')