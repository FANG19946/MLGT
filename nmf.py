import numpy as np
import torch
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
from skmultilearn.dataset import load_from_arff


# =========================================================
# DATA
# =========================================================
def load_data():
    train_path = "datasets/bibtex/bibtex-train.arff"
    test_path = "datasets/bibtex/bibtex-test.arff"

    X_train_full, Y_train_full = load_from_arff(
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

    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train_full,
        Y_train_full,
        test_size=0.2,
        random_state=42
    )

    def fix(X, Y):
        X = X.toarray() if hasattr(X, "toarray") else X
        Y = Y.toarray() if hasattr(Y, "toarray") else Y
        return np.asarray(X, np.float32), np.asarray(Y, np.float32)

    return (*fix(X_train, Y_train),
            *fix(X_val, Y_val),
            *fix(X_test, Y_test))


# =========================================================
# SYMNMF (LABEL CORRELATION FACTORIZATION)
# =========================================================
def symnmf(Y, m, seed=0):
    C = (Y.T @ Y).astype(np.float32)

    model = NMF(n_components=m, init="random", random_state=seed, max_iter=300)
    W = model.fit_transform(C)
    H = model.components_

    return H  # (m × d)


# =========================================================
# BUILD GROUP MATRIX A
# =========================================================
def build_A(H, c=6, seed=0):
    rng = np.random.default_rng(seed)

    m, d = H.shape
    A = np.zeros((m, d), dtype=bool)

    # normalize columns of H → probabilities
    H = np.maximum(H, 0)
    H = H / (H.sum(axis=0, keepdims=True) + 1e-12)

    # track row usage (IMPORTANT)
    row_load = np.zeros(m)

    target_row_load = (c * d) / m

    for j in range(d):
        prob = H[:, j]

        # avoid collapse: mix randomness + structure
        prob = 0.7 * prob + 0.3 * (1.0 / m)

        # prevent overloading rows
        prob = prob / (row_load + 1e-3)

        prob = np.maximum(prob, 0)
        prob = prob / (prob.sum() + 1e-12)

        chosen = rng.choice(m, size=c, replace=False, p=prob)

        A[chosen, j] = True
        row_load[chosen] += 1

    # -------------------------------------------------
    # FIX: guarantee no empty rows
    # -------------------------------------------------
    row_sums = A.sum(axis=1)

    empty_rows = np.where(row_sums == 0)[0]

    for i in empty_rows:
        j = rng.integers(0, d)
        A[i, j] = True

    return A
# =========================================================
# TRAINING
# =========================================================
def train(X, Y, A, epochs=30):
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)

    n, d = Y.shape
    m = A.shape[0]

    # =====================================================
    # group labels (FIXED: no astype, no tensor/numpy mix)
    # =====================================================
    Yg = np.zeros((X.shape[0], m), dtype=np.float32)

    for i in range(m):
        mask = A[i]

        if np.sum(mask) == 0:
            continue

        # correct PyTorch operation
        Yg[:, i] = (Y[:, mask].sum(dim=1) > 0).cpu().numpy()

    Yg = torch.tensor(Yg, dtype=torch.float32)

    # =====================================================
    # classifier
    # =====================================================
    W = torch.randn(m, X.shape[1], requires_grad=True)
    opt = torch.optim.Adam([W], lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for ep in range(epochs):
        logits = X @ W.T

        loss = loss_fn(logits, Yg)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if ep % 5 == 0:
            print(f"epoch {ep} | loss {loss.item():.4f}")

    return W.detach()


# =========================================================
# DECODER (GROUP → LABELS)
# =========================================================
def decode(A, group_pred):
    group_pred = group_pred.astype(int)

    # how many groups activate each label
    scores = A.T @ group_pred

    # threshold (paper-style voting)
    return (scores >= 2).astype(int)


# =========================================================
# EVALUATION
# =========================================================
def evaluate(X, Y, W, A, k):
    X = torch.tensor(X, dtype=torch.float32)

    logits = X @ W.T
    probs = torch.sigmoid(logits).detach().numpy()

    preds = []

    for i in range(len(X)):
        group_pred = (probs[i] > 0.5)
        label_pred = decode(A, group_pred)
        preds.append(label_pred)

    preds = np.array(preds)

    hamming = np.mean(preds != Y)

    precision = []
    for i in range(len(Y)):
        true = np.where(Y[i] == 1)[0]
        topk = np.argsort(-preds[i])[:k]

        if len(true):
            precision.append(len(set(true) & set(topk)) / k)

    return {
        "hamming_loss": hamming,
        "precision@k": np.mean(precision) if precision else 0.0
    }


# =========================================================
# RUN
# =========================================================
def run():
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_data()

    k = max(1, int(Y_train.sum(axis=1).mean()))
    m = 100

    print("\nBuilding SymNMF...")
    H = symnmf(Y_train, m)

    print("Building A...")
    A = build_A(H, c=4)

    print("\n--- LABEL ASSIGNMENT MATRIX (A) DEBUG ---")
    # =====================================================
    # EMPTY ROW DEBUG
    # =====================================================
    row_sums = A.sum(axis=1)

    num_empty_rows = np.sum(row_sums == 0)
    num_near_empty = np.sum(row_sums <= 1)

    print("\n--- ROW SPARSITY DEBUG ---")
    print("Total rows:", A.shape[0])
    print("Empty rows:", num_empty_rows)
    print("Near-empty rows (<=1):", num_near_empty)
    print("Avg row degree:", row_sums.mean())
    print("Max row degree:", row_sums.max())
    print("Min row degree:", row_sums.min())
    print("Shape:", A.shape)
    print("Avg ones per column (should be ~c):", A.sum(axis=0).mean())
    print("Avg ones per row:", A.sum(axis=1).mean())
    print("\nSample rows (groups → labels):")

    print("\nSample columns (labels → groups):")

    for j in range(min(5, A.shape[1])):
        groups = np.where(A[:, j])[0]
        print(f"Label {j}: {groups}")

    for i in range(min(5, A.shape[0])):
        labels = np.where(A[i])[0]
        print(f"Group {i}: {labels}")

    print("Training...")
    W = train(X_train, Y_train, A)

    print("\nTEST")
    print(evaluate(X_test, Y_test, W, A, 4))


if __name__ == "__main__":
    run()