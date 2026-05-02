import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from skmultilearn.dataset import load_from_arff
from sklearn.model_selection import train_test_split
import os
from skmultilearn.dataset import load_from_arff
import arff
from sklearn.preprocessing import MultiLabelBinarizer

# Dataset loaders

def parse_libsvm_multilabel(path, max_feat=5000):
    X_list = []
    Y_list = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()

            
            raw_labels = parts[0].split(",")

            labels = []
            for x in raw_labels:
               
                if ":" in x:
                    continue
                try:
                    labels.append(int(x))
                except:
                    continue

            Y_list.append(labels)

           
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

    print("\nEURLEX LOADING")

    X_train, Y_train_list = parse_libsvm_multilabel(train_path)
    X_test,  Y_test_list  = parse_libsvm_multilabel(test_path)

    
    mlb = MultiLabelBinarizer()

    # IMPORTANT FIX: avoid "unknown classes" warning
    mlb.fit(Y_train_list + Y_test_list)

    Y_train = mlb.transform(Y_train_list)
    Y_test  = mlb.transform(Y_test_list)

   
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train, Y_train,
        test_size=0.2,
        random_state=42
    )

    # print("\n[EURLEX FIXED LOADER]")
    # print("X_train:", X_train.shape)
    # print("X_val:", X_val.shape)
    # print("X_test:", X_test.shape)

    # print("Y_train:", Y_train.shape)
    # print("Y_val:", Y_val.shape)
    # print("Y_test:", Y_test.shape)

    # print("n_labels:", Y_train.shape[1])
    # print("avg labels/sample:", Y_train.sum(axis=1).mean())

    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)

def load_bibtex():
    path = "datasets/bibtex"

    train_path = os.path.join(path, "bibtex-train.arff")
    test_path = os.path.join(path, "bibtex-test.arff")

    X_train, Y_train = load_from_arff(
        train_path,
        label_count=159,  
        label_location="end",
        load_sparse=False
    )

    X_test, Y_test = load_from_arff(
        test_path,
        label_count=159,  
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



# Testing Matrices (Label Assignment)

def bernoulli_matrix(d, m, k_target, c=3, seed=0):
    rng = np.random.default_rng(seed)
    p = c / k_target
    return (rng.random((m, d)) < p).astype(bool)

# def bernoulli_matrix(d, m, w, seed=0):
#     rng = np.random.default_rng(seed)
#     A = np.zeros((m, d), dtype=bool)

#     for j in range(d):
#         rows = rng.choice(m, size=w, replace=False)
#         A[rows, j] = True

#     return A

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




class BinaryClassifier(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Linear(d, 1)

    def forward(self, x):
        return self.net(x).squeeze(-1)



def train_models(X, Y, A, epochs=40, lr=1e-3, device="cuda"):

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





def predict(models, X, device="cuda"):

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

    return np.stack(preds, axis=1)  




def decode(A, scores, k):
    label_scores = A.T @ scores
    return np.argsort(-label_scores)[:k]




def evaluate(models, X, Y, A, k_eval=4, k_decode=None, device="cuda"):

    probs = predict(models, X, device)

    if hasattr(Y, "toarray"):
        Y = Y.toarray()

    Y = np.array(Y, dtype=int)

    n_samples, n_labels = Y.shape
    n_tests = A.shape[0] 
    pred = np.zeros((n_samples, n_labels), dtype=int)
    if k_decode is None:
                k_decode = int(np.mean(Y.sum(axis=1)))  
    for i in range(n_samples):
        

        topk = decode(A, probs[i], k_decode)
        pred[i, topk] = 1

    
    hl = np.mean(pred != Y)

    
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




def run():

    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = load_eurlex()

    n_labels = Ytr.shape[1]
    m = int(10 * np.log(n_labels))
    

    print("n_labels:", n_labels)
    print("n_tests:", m)

    k_target = int(np.mean(Ytr.sum(axis=1))) + 1


    constructions = {
        "bernoulli": lambda: bernoulli_matrix(n_labels, m, k_target),
        "expander": lambda: expander_matrix(n_labels, m),
        # "identity": lambda: identity_matrix(n_labels)
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