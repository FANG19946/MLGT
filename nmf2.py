import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from skmultilearn.dataset import load_from_arff
from sklearn.model_selection import train_test_split
from scipy.sparse import issparse

# =========================================================
# DATA
# =========================================================
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import numpy as np
def parse_libsvm_multilabel(path, max_feat=5000):
    X_list = []
    Y_list = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()

            # -------------------------
            # LABEL PARSING (FIRST TOKEN)
            # -------------------------
            raw_labels = parts[0].split(",")

            labels = []
            for x in raw_labels:
                # safety: skip corrupted tokens like "0:0.08"
                if ":" in x:
                    continue
                try:
                    labels.append(int(x))
                except:
                    continue

            Y_list.append(labels)

            # -------------------------
            # FEATURE PARSING
            # -------------------------
            x = np.zeros(max_feat, dtype=np.float32)

            for item in parts[1:]:
                if ":" not in item:
                    continue
                try:
                    idx, val = item.split(":")
                    idx = int(idx)
                    if idx < max_feat:
                        x[idx] = float(val)
                except:
                    continue

            X_list.append(x)

    return np.array(X_list, dtype=np.float32), Y_list

def load_eurlex():

    train_path = "datasets/eurlex/eurlex_train.txt"
    test_path  = "datasets/eurlex/eurlex_test.txt"

    print("\n[EURLEX LOADING...]")

    X_train, Y_train_list = parse_libsvm_multilabel(train_path)
    X_test,  Y_test_list  = parse_libsvm_multilabel(test_path)

    mlb = MultiLabelBinarizer()
    mlb.fit(Y_train_list + Y_test_list)

    Y_train = mlb.transform(Y_train_list)
    Y_test  = mlb.transform(Y_test_list)

    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train, Y_train,
        test_size=0.2,
        random_state=42
    )

    def fix(X, Y):
        X = X.toarray() if hasattr(X, "toarray") else X
        Y = Y.toarray() if hasattr(Y, "toarray") else Y
        return np.asarray(X, np.float32), np.asarray(Y, np.float32)

    Xtr, Ytr = fix(X_train, Y_train)
    Xva, Yva = fix(X_val, Y_val)
    Xte, Yte = fix(X_test, Y_test)

    print("\n[EURLEX FIXED LOADER]")
    print("X_train:", Xtr.shape)
    print("X_val:", Xva.shape)
    print("X_test:", Xte.shape)
    print("Y_train:", Ytr.shape)
    print("Y_val:", Yva.shape)
    print("Y_test:", Yte.shape)

    print("n_labels:", Ytr.shape[1])
    print("avg labels/sample:", Ytr.sum(axis=1).mean())

    return Xtr, Ytr, Xte, Yte


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

    X_train, _, Y_train, _ = train_test_split(
        X_train, Y_train, test_size=0.2, random_state=42
    )

    def fix(X, Y):
        if hasattr(X, "toarray"):
            X = X.toarray()
        if hasattr(Y, "toarray"):
            Y = Y.toarray()
        return np.asarray(X, np.float32), np.asarray(Y, np.float32)

    return fix(X_train, Y_train) + fix(X_test, Y_test)


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

    X_train, _, Y_train, _ = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

    return X_train, Y_train, X_test, Y_test


# =========================================================
# SAFE CONVERSION
# =========================================================

def to_dense(X):
    if issparse(X):
        return X.toarray()
    return np.array(X)


# =========================================================
# SYMNMF (simple multiplicative updates)
# =========================================================

def symnmf(S, r=10, steps=200, lr=0.01):
    n = S.shape[0]
    H = np.abs(np.random.randn(n, r))

    for i in range(steps):
        HS = H @ H.T @ H
        H = H * (S @ H) / (HS + 1e-8)

        if i % 20 == 0:
            loss = np.linalg.norm(S - H @ H.T)
            print(f"[SymNMF] iter {i} loss {loss:.4f}")

    return H


# =========================================================
# BUILD LABEL GRAPH
# =========================================================

def build_S(Y):
    Y = to_dense(Y).astype(float)
    return Y.T @ Y


# =========================================================
# CLUSTER LABELS
# =========================================================

def cluster_labels(H):
    return np.argmax(H, axis=1)


def build_groups(H):
    clusters = cluster_labels(H)
    r = H.shape[1]

    groups = []
    for i in range(r):
        groups.append(np.where(clusters == i)[0])

    return groups


# =========================================================
# MODEL (one per group)
# =========================================================

class GroupClassifier(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Linear(d, 1)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# =========================================================
# TRAIN GROUP MODELS
# =========================================================

def train_group_models(X, Y, groups, epochs=20, lr=1e-3):
    X = torch.tensor(to_dense(X), dtype=torch.float32)
    Y = to_dense(Y).astype(float)

    models = []
    d = X.shape[1]

    for g in groups:

        if len(g) == 0:
            models.append(None)
            continue

        Yg = (Y[:, g].sum(axis=1) > 0).astype(float)
        Yg = torch.tensor(Yg, dtype=torch.float32)

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
    X = torch.tensor(to_dense(X), dtype=torch.float32)

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

def decode(groups, scores, k):
    label_scores = np.zeros(scores.shape[0])

    for i, g in enumerate(groups):
        label_scores[g] += scores[i]

    return np.argsort(-label_scores)[:k]


# =========================================================
# EVALUATION
# =========================================================

def evaluate(models, X, Y, groups, k=4):

    probs = predict(models, X)

    Y = to_dense(Y).astype(int)

    n_samples, n_labels = Y.shape
    
    n_tests = len(models)
    pred = np.zeros_like(Y)

    for i in range(n_samples):
        topk = decode(groups, probs[i], k, n_labels)
        pred[i, topk] = 1

    hl = np.mean(pred != Y)

    p_scores = []

    for i in range(n_samples):
        true = np.where(Y[i] == 1)[0]
        if len(true) == 0:
            continue

        label_scores = groups_to_label_scores(groups, probs[i], n_labels)
        predk = np.argsort(-label_scores)[:k]

        p_scores.append(len(set(predk) & set(true)) / k)

    return {
        "hamming_loss": hl,
        "precision@k": np.mean(p_scores),
        "n_labels": n_labels,
        "n_tests": n_tests
    }

def groups_to_label_scores(groups, scores, n_labels):
    label_scores = np.zeros(n_labels)

    for i, g in enumerate(groups):
        if len(g) == 0:
            continue
        label_scores[g] += scores[i]

    return label_scores


def decode(groups, scores, k, n_labels):
    label_scores = groups_to_label_scores(groups, scores, n_labels)
    return np.argsort(-label_scores)[:k]


# =========================================================
# RUN
# =========================================================

def run():

    Xtr, Ytr, Xte, Yte = load_eurlex()

    print("labels:", Ytr.shape[1])

    S = build_S(Ytr)

    H = symnmf(S, r=82, steps=200)

    groups = build_groups(H)

    models = train_group_models(Xtr, Ytr, groups)

    metrics = evaluate(models, Xte, Yte, groups, k=5)

    print("\n===== RESULTS =====")
    print("\n===== RESULTS =====")
    print(f"hamming_loss: {metrics['hamming_loss']:.3f}")
    print(f"precision@k: {metrics['precision@k']:.3f}")
    print(f"n_labels: {metrics['n_labels']}")
    print(f"n_tests: {metrics['n_tests']}")


if __name__ == "__main__":
    run()