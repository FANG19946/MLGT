import numpy as np
from scipy.sparse import csr_matrix
from bitarray import bitarray
import torch


def build_testing_matrix(
        n_labels,
        k,
        method='bernoulli',
        seed=None
        ):
    match method:
        case 'bernoulli':
            p = 1/(k+1)
            n_tests = int(4 * k * np.log(n_labels))
            rng = np.random.default_rng(seed)
            A = rng.random((n_tests, n_labels)) < p
            e = int(0.5 * k )

            
        
        case 'rs':
            rng = np.random.default_rng(seed)

            q = 16
            L = int(np.ceil(np.log(n_labels) / np.log(q)))
            n_tests = q * L

            e = L - 1   # standard RS bound

            A = np.zeros((n_tests, n_labels), dtype=bool)

            messages = rng.integers(0, q, size=(n_labels, L))

            for j in range(n_labels):
                for i in range(L):
                    sym = (messages[j, i] + i * (j + 1)) % q
                    row = i * q + sym
                    A[row, j] = True

            
        
        case 'expander':
            rng = np.random.default_rng(seed)

            # left degree 
            ell = int(4 * np.log(n_labels))
            n_tests = int(  k * k * np.log(n_labels))

            A = np.zeros((n_tests, n_labels), dtype=bool)

            
            load = np.zeros(n_tests, dtype=int)

            for j in range(n_labels):
                chosen = set()

                
                while len(chosen) < ell:
                    i = rng.integers(0, n_tests)
                    chosen.add(i)

                for i in chosen:
                    A[i, j] = True
                    load[i] += 1

            e = ell // 4

        case 'identity':
            n_tests = n_labels
            A = np.eye(n_labels, dtype=bool)
            e=0

           
            

    n_tests = A.shape[0]
    A_bit = []
    for i in range(A.shape[0]):
        ba = bitarray()
        ba.extend(A[i].tolist())   
        A_bit.append(ba)

    A = A_bit

    
    return A,n_tests,e

def decoder(A, y, e):
    A = np.array([row.tolist() for row in A], dtype=bool)
    y = np.array(y, dtype=bool)
    
    
    n_tests, n_labels = A.shape

    z = np.zeros(n_labels, dtype=int)

    supp_y = set(np.where(y)[0])          
    for i in range(n_labels):

        supp_A_i = set(np.where(A[:, i])[0])   

        missing = supp_A_i - supp_y           

        if len(missing) < e // 2:
            z[i] = 1
        else:
            z[i] = 0
    return z

def dataset_params(trainset, e_ratio):
    if hasattr(trainset, "toarray"):
        trainset = trainset.toarray()

    trainset = trainset.astype(bool)
    
    n_tests, n_labels = trainset.shape

    sample_sparsity = trainset.sum(axis=1)
    label_frequency = trainset.sum(axis=0)
    mean_frequency = np.mean(label_frequency)
    k=int(np.median(sample_sparsity))
    

    return n_labels,n_tests,k,None

def train_classifiers(dataset, A, epochs, lr=0.001, device='cuda'):
    feature_matrix, label_matrix = dataset
    
    # convert sparse → dense if needed
    if hasattr(feature_matrix, "toarray"):
        feature_matrix = feature_matrix.toarray()

    if hasattr(label_matrix, "toarray"):
        label_matrix = label_matrix.toarray()

    feature_matrix = torch.tensor(feature_matrix, dtype=torch.float32, device=device)
    label_matrix = torch.tensor(label_matrix, dtype=torch.float32, device=device)
    
    A = torch.tensor(A, dtype=torch.bool, device=device)

    n_samples, d = feature_matrix.shape
    n_labels = label_matrix.shape[1]
    n_tests = A.shape[0]

    Y_test = torch.zeros((n_samples, n_tests), device=device)

    for i in range(n_tests):
        label_mask = A[i]  
        Y_test[:, i] = (label_matrix[:, label_mask].sum(dim=1) > 0)

    Y_test = Y_test.float()
    
    W = torch.randn(n_tests, d, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([W], lr=lr)
    
    pos = Y_test.sum(dim=0)
    neg = Y_test.shape[0] - pos

    pos_weight = neg / (pos + 1e-6)
    pos_weight = torch.clamp(pos_weight, min=1.0, max=10.0).to(device)

    pos_weight = pos_weight.to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    for _ in range(epochs):

        logits = feature_matrix @ W.T   # (n_samples, n_tests)

        loss = loss_fn(logits, Y_test)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return W.detach()

def evaluation_metrics(W, dataset, A, k,e,threshold=0.1, device='cuda'):
    feature_matrix, label_matrix = dataset
    if hasattr(feature_matrix, "toarray"):
        feature_matrix = feature_matrix.toarray()

    if hasattr(label_matrix, "toarray"):
        label_matrix = label_matrix.toarray()

    feature_matrix = torch.tensor(feature_matrix, dtype=torch.float32, device=device)
    label_matrix = torch.tensor(label_matrix, dtype=torch.int32, device=device)
    A = torch.tensor(A, dtype=torch.bool, device=device)

    n_samples = feature_matrix.shape[0]
    n_tests = A.shape[0]
    n_labels = A.shape[1]


    logits = feature_matrix @ W.T
    result = torch.sigmoid(logits)
    probs = result.detach().cpu().numpy()
    result = (result > threshold)

    result=result.cpu().numpy().astype(bool)
    A = A.cpu().numpy().astype(bool)
    label_matrix = label_matrix.cpu().numpy()

    if n_tests == n_labels:
        # identity matrix → no decoding needed
        full_result = result.astype(int)
    else:
        full_result = np.zeros((n_samples, n_labels), dtype=int)
        for i in range(n_samples):
            full_result[i] = decoder(A, result[i], e)



    true_result = label_matrix
    # Hamming loss
    hamming_loss = np.mean(true_result != full_result)

    print("\n--- Decoder Debug ---")
    # print("Avg predicted labels per sample:", full_result.sum(axis=1).mean())
    avg_pred_label_per_sample = full_result.sum(axis=1).mean()
    print("Min predicted labels:", full_result.sum(axis=1).min())
    print("Max predicted labels:", full_result.sum(axis=1).max())

    # print("Avg true labels per sample:", true_result.sum(axis=1).mean())

    # Compare one example
    idx = 0
    print("\nExample sample:")
    print("Predicted labels:", np.where(full_result[idx] == 1)[0][:20])
    print("True labels:", np.where(true_result[idx] == 1)[0])


    precision_scores = []

    A_np = A  # already numpy bool

    for i in range(n_samples):

        z = probs[i]

        # compute label scores
        scores = A_np.T @ z         # (n_labels,)

        top_k = np.argsort(-scores)[:k]

        true_labels = np.where(true_result[i] == 1)[0]

        if len(true_labels) == 0:
            continue

        hits = len(set(top_k) & set(true_labels))
        precision_scores.append(hits / k)

    precision_at_k = np.mean(precision_scores) if precision_scores else 0.0


    return {
    "hamming_loss": hamming_loss,
    "precision@k": precision_at_k,
    "n_labels": n_labels,
    "n_tests": n_tests,
    "avg_pred": avg_pred_label_per_sample
    }


