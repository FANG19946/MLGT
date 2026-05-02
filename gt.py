import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from skmultilearn.dataset import load_from_arff
from sklearn.model_selection import train_test_split
import os

# =========================================================
# DATA LOADING (MEDIAMILL)
# =========================================================

import os
from skmultilearn.dataset import load_from_arff
from sklearn.model_selection import train_test_split


import os
from skmultilearn.dataset import load_from_arff
from sklearn.model_selection import train_test_split

import os
import numpy as np
from sklearn.model_selection import train_test_split
from skmultilearn.dataset import load_from_arff


from mldr.datasets import get_mldr

def load_wiki10():
    data = get_mldr("Wiki10-31K")

    X = data["data"]
    Y = data["target"]

    from sklearn.model_selection import train_test_split

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    print("[Wiki10] labels:", Y.shape[1])

    return (X_train, Y_train), (None, None), (X_test, Y_test)


def load_bibtex():
    path = "datasets/bibtex"

    train_path = os.path.join(path, "bibtex-train.arff")
    test_path = os.path.join(path, "bibtex-test.arff")

    X_train, Y_train = load_from_arff(
        train_path,
        label_count=159,   # ✅ FIXED
        label_location="end",
        load_sparse=False
    )

    X_test, Y_test = load_from_arff(
        test_path,
        label_count=159,   # ✅ FIXED
        label_location="end",
        load_sparse=False
    )

    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train, Y_train, test_size=0.2, random_state=42
    )

    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)

def load_mediamill():
    path = "datasets/mediamill"

    train_path = os.path.join(path, "mediamill-train.arff")
    test_path = os.path.join(path, "mediamill-test.arff")

    X_train, Y_train = load_from_arff(
        train_path, label_count=101, label_location="end", load_sparse=False
    )
    X_test, Y_test = load_from_arff(
        test_path, label_count=101, label_location="end", load_sparse=False
    )

    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train, Y_train, test_size=0.2, random_state=42
    )

    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)


# =========================================================
# MATRIX GENERATORS
# =========================================================

# def bernoulli_matrix(d, m, seed=0, p=None):
#     rng = np.random.default_rng(seed)
#     if p is None:
#         p = 1 / np.sqrt(d)
#     return (rng.random((m, d)) < p).astype(bool)

def bernoulli_matrix(d, m, k_target, c=3, seed=0):
    rng = np.random.default_rng(seed)
    p = c / k_target
    return (rng.random((m, d)) < p).astype(bool)


def identity_matrix(d):
    return np.eye(d, dtype=bool)


def expander_matrix(d, m, left_deg=None, seed=0):
    rng = np.random.default_rng(seed)
    if left_deg is None:
        left_deg = int(np.log(d) + 2)

    A = np.zeros((m, d), dtype=bool)

    for j in range(d):
        rows = rng.choice(m, size=left_deg, replace=False)
        A[rows, j] = True

    return A


# =========================================================
# MODEL
# =========================================================

class BinaryClassifier(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Linear(d, 1)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# =========================================================
# TRAINING (INDEPENDENT CLASSIFIERS)
# =========================================================

def train_models(X, Y, A, epochs=40, lr=1e-3, device="cpu"):

    if hasattr(X, "toarray"):
        X = X.toarray()
    if hasattr(Y, "toarray"):
        Y = Y.toarray()

    X = torch.tensor(X, dtype=torch.float32, device=device)
    Y = torch.tensor(Y, dtype=torch.float32, device=device)

    n_tests = A.shape[0]
    d = X.shape[1]

    models = []

    for j in range(n_tests):

        cols = np.where(A[j])[0]

        if len(cols) == 0:
            models.append(None)
            continue

        # FIX: flatten (this was your bug)
        Yj = (Y[:, cols].sum(axis=1) > 0).float().ravel()

        model = BinaryClassifier(d).to(device)
        opt = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()

        for _ in range(epochs):
            logits = model(X)
            loss = loss_fn(logits, Yj)

            opt.zero_grad()
            loss.backward()
            opt.step()

        models.append(model)

    return models


# =========================================================
# PREDICTION
# =========================================================

def predict(models, X, device="cpu"):

    if hasattr(X, "toarray"):
        X = X.toarray()

    X = torch.tensor(X, dtype=torch.float32, device=device)

    preds = []

    with torch.no_grad():
        for m in models:
            if m is None:
                preds.append(np.zeros(len(X)))
            else:
                preds.append(torch.sigmoid(m(X)).cpu().numpy())

    return np.stack(preds, axis=1)  # (n_samples, n_tests)


# =========================================================
# DECODING (MLGT STYLE)
# =========================================================

def decode(A, scores, k):
    label_scores = A.T @ scores
    return np.argsort(-label_scores)[:k]


# =========================================================
# EVALUATION (CLEAN + STABLE)
# =========================================================

def evaluate(models, X, Y, A, k_eval=4, k_decode=None, device="cpu"):

    probs = predict(models, X, device)

    if hasattr(Y, "toarray"):
        Y = Y.toarray()

    Y = np.array(Y, dtype=int)

    n_samples, n_labels = Y.shape
    n_tests = A.shape[0] 
    pred = np.zeros((n_samples, n_labels), dtype=int)
    if k_decode is None:
                k_decode = int(np.mean(Y.sum(axis=1)))  # dataset sparsity estimate
    for i in range(n_samples):
        

        topk = decode(A, probs[i], k_decode)
        pred[i, topk] = 1

    # HL
    hl = np.mean(pred != Y)

    # P@k (clean ranking version)
    p_scores = []

    for i in range(n_samples):
        true = np.where(Y[i] == 1)[0]
        if len(true) == 0:
            continue

        ranked = np.argsort(-(A.T @ probs[i]))[:k_eval]
        p_scores.append(len(set(ranked) & set(true)) / k_eval)

    return {
        "hamming_loss": hl,
        "precision@k": np.mean(p_scores),
         "n_tests": n_tests 
    }


# =========================================================
# RUN
# =========================================================

# def run():

#     (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = load_mediamill()

#     n_labels = Ytr.shape[1]
#     m = int(10 * np.log(n_labels + 1))

#     print("n_labels:", n_labels)
#     print("n_tests:", m)

#     A = bernoulli_matrix(n_labels, m)

#     models = train_models(Xtr, Ytr, A, epochs=20)

#     metrics = evaluate(models, Xte, Yte, A, k=4)

#     print("\n===== RESULT =====")
#     print(metrics)


def run():

    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = load_wiki10()

    n_labels = Ytr.shape[1]
    m = int(10 * np.log(n_labels + 1))
    

    print("n_labels:", n_labels)
    print("n_tests:", m)

    k_target = int(np.mean(Ytr.sum(axis=1))) + 1


    constructions = {
        "bernoulli": lambda: bernoulli_matrix(n_labels, m, k_target),
        "expander": lambda: expander_matrix(n_labels, m),
        "identity": lambda: identity_matrix(n_labels)
    }

    results = {}

    for name, build_A in constructions.items():

        print(f"\n--- {name} ---")

        A = build_A()

        models = train_models(Xtr, Ytr, A, epochs=20)

        metrics = evaluate(models, Xte, Yte, A, k_eval=4)

        results[name] = metrics

        print(metrics)

    print("\n===== FINAL SUMMARY =====")
    for k_, v in results.items():
        print(k_, v)

if __name__ == "__main__":
    run()