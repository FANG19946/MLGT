import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from skmultilearn.dataset import load_from_arff
from sklearn.model_selection import train_test_split
from scipy.sparse import issparse


# =========================================================
# DATA (MEDIAMILL)
# =========================================================
def load_bibtex():
    path = "datasets/bibtex"

    X_train, Y_train = load_from_arff(
        path + "/bibtex-train.arff",
        label_count=159,
        label_location="end",
        load_sparse=False
    )

    X_test, Y_test = load_from_arff(
        path + "/bibtex-test.arff",
        label_count=159,
        label_location="end",
        load_sparse=False
    )

    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train, Y_train, test_size=0.2, random_state=42
    )

    def fix(X, Y):
        X = X.toarray() if hasattr(X, "toarray") else X
        Y = Y.toarray() if hasattr(Y, "toarray") else Y
        return np.asarray(X, np.float32), np.asarray(Y, np.float32)

    Xtr, Ytr = fix(X_train, Y_train)
    Xva, Yva = fix(X_val, Y_val)
    Xte, Yte = fix(X_test, Y_test)

    return Xtr, Ytr, Xva, Yva, Xte, Yte

def load_mediamill():
    path = "datasets/mediamill"

    X_train, Y_train = load_from_arff(
        path + "/mediamill-train.arff",
        label_count=101,
        label_location="end",
        load_sparse=False
    )

    X_test, Y_test = load_from_arff(
        path + "/mediamill-test.arff",
        label_count=101,
        label_location="end",
        load_sparse=False
    )

    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train, Y_train, test_size=0.2, random_state=42
    )

    def fix(X, Y):
        X = X.toarray() if hasattr(X, "toarray") else X
        Y = Y.toarray() if hasattr(Y, "toarray") else Y
        return np.asarray(X, np.float32), np.asarray(Y, np.float32)

    return (*fix(X_train, Y_train),
            *fix(X_val, Y_val),
            *fix(X_test, Y_test))


# =========================================================
# YOUR STRUCTURED A MATRIX (ALPHABET / K-S STYLE)
# =========================================================
def build_A(n_labels, seed=0):
    rng = np.random.default_rng(seed)

    q = 16  # alphabet size
    L = int(np.ceil(np.log(n_labels) / np.log(q)))
    m = q * L

    A = np.zeros((m, n_labels), dtype=bool)

    messages = rng.integers(0, q, size=(n_labels, L))

    for j in range(n_labels):
        for i in range(L):

            sym = (messages[j, i] + i * (j + 1)) % q
            row = i * q + sym

            A[row, j] = True

    print("\n[A MATRIX]")
    print("labels:", n_labels)
    print("groups:", m)
    print("avg label degree:", A.sum(axis=0).mean())
    print("avg group size:", A.sum(axis=1).mean())

    return A


# =========================================================
# GROUP MODELS
# =========================================================
class GroupClassifier(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Linear(d, 1)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# =========================================================
# TRAIN
# =========================================================
def train(X, Y, A, epochs=20, lr=1e-3):

    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)

    n_labels = Y.shape[1]
    m = A.shape[0]

    models = []
    d = X.shape[1]

    for i in range(m):

        labels = np.where(A[i])[0]
        if len(labels) == 0:
            models.append(None)
            continue

        Yg = (Y[:, labels].sum(dim=1) > 0).float()

        model = GroupClassifier(d)
        opt = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()

        for _ in range(epochs):
            logits = model(X)
            loss = loss_fn(logits, Yg)

            opt.zero_grad()
            loss.backward()
            opt.step()

        models.append(model)

    return models


# =========================================================
# PREDICT
# =========================================================
def predict(models, X):
    X = torch.tensor(X, dtype=torch.float32)

    outs = []
    with torch.no_grad():
        for m in models:
            if m is None:
                outs.append(np.zeros(len(X)))
            else:
                outs.append(torch.sigmoid(m(X)).numpy())

    return np.stack(outs, axis=1)


# =========================================================
# DECODE
# =========================================================
def decode(A, group_scores, k):
    label_scores = A.T @ group_scores
    return np.argsort(-label_scores)[:k]


# =========================================================
# EVAL
# =========================================================
def evaluate(models, X, Y, A, k=1):

    probs = predict(models, X)

    Y = Y.astype(int)

    n = Y.shape[0]

    pred = np.zeros_like(Y)

    for i in range(n):
        topk = decode(A, probs[i], k)
        pred[i, topk] = 1

    hl = np.mean(pred != Y)

    p_scores = []
    for i in range(n):
        true = np.where(Y[i] == 1)[0]
        if len(true) == 0:
            continue

        label_scores = A.T @ probs[i]
        topk = np.argsort(-label_scores)[:k]

        p_scores.append(len(set(topk) & set(true)) / k)

    return {
        "hamming_loss": hl,
        "precision@k": np.mean(p_scores)
    }


# =========================================================
# RUN
# =========================================================
def run():

    Xtr, Ytr, Xva, Yva, Xte, Yte = load_bibtex()

    n_labels = Ytr.shape[1]
    n_tests = Xte.shape[0]

    print("\nlabels:", n_labels)
    print("n_tests:", n_tests)

    print("\nlabels:", Ytr.shape[1])

    A = build_A(Ytr.shape[1], seed=42)

    n_labels = A.shape[1]
    n_tests = A.shape[0]

    print("\nlabels:", n_labels)
    print("n_tests:", n_tests)
    
    print("\nlabels:", Ytr.shape[1])


    print("\ntraining...")

    models = train(Xtr, Ytr, A)

    print("\nEVAL")

    metrics = evaluate(models, Xte, Yte, A, k=5)

    print(metrics)


if __name__ == "__main__":
    run()