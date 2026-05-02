import numpy as np
from scipy.sparse import csr_matrix
from bitarray import bitarray
import torch


def build_testing_matrix(
        n_tests,n_labels,
        k,e=0,
        method='bernoulli',
        seed=None
        ):
    match method:
        case 'bernoulli':
            p = 1/(k+1)
            rng = np.random.default_rng(seed)
            A = rng.random((n_tests, n_labels)) < p

            
        
        case 'rs':
            rng = np.random.default_rng(seed)

            # alphabet size 
            q = 16


            L = int(np.ceil(np.log(n_labels) / np.log(q)))

            m = q * L

            A = np.zeros((m, n_labels), dtype=bool)

            
            messages = rng.integers(0, q, size=(n_labels, L))

            for j in range(n_labels):
                for i in range(L):

                    sym = (messages[j, i] + i * (j + 1)) % q

                    row = i * q + sym
                    A[row, j] = True

            
        
        case 'expander':
            rng = np.random.default_rng(seed)

            # left degree 
            ell = max(3, int(np.log(n_labels)))

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

        case 'identity':
            n_tests = n_labels
            A = np.eye(n_labels, dtype=bool)

           
            


    A_bit = []
    for i in range(A.shape[0]):
        ba = bitarray()
        ba.extend(A[i].tolist())   
        A_bit.append(ba)

    A = A_bit

    
    return A




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
    trainset = np.array(trainset,dtype=bool)
    n_tests, n_labels = trainset.shape

    sample_sparsity = trainset.sum(axis=1)
    label_frequency = trainset.sum(axis=0)
    mean_frequency = np.mean(label_frequency)
    k=int(np.median(sample_sparsity))
    e=max(1, int(e_ratio * mean_frequency))

    return n_labels,n_tests,k,e

def train_classifiers(dataset, A, epochs, lr=0.001, device='cuda'):
    feature_matrix, label_matrix = dataset
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

    
    W = torch.randn(n_tests, d, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([W], lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for _ in range(epochs):

        logits = feature_matrix @ W.T   # (n_samples, n_tests)

        loss = loss_fn(logits, Y_test)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return W.detach()

def evaluation_metrics(W, dataset, A, k,e, device='cuda'):
    feature_matrix, label_matrix = dataset

    feature_matrix = torch.tensor(feature_matrix, dtype=torch.float32, device=device)
    label_matrix = torch.tensor(label_matrix, dtype=torch.int32, device=device)
    A = torch.tensor(A, dtype=torch.bool, device=device)

    n_samples = feature_matrix.shape[0]
    n_tests = A.shape[0]
    n_labels = A.shape[1]


    logits = feature_matrix @ W.T
    result = torch.sigmoid(logits)
    result = (result > 0.5)

    result=result.cpu().numpy().astype(bool)
    A = A.cpu().numpy().astype(bool)
    label_matrix = label_matrix.cpu().numpy()

    full_result = np.zeros((n_samples, n_labels), dtype=int)
    for i in range(n_samples):
        full_result[i] = decoder(A, result[i], e)

    true_result = label_matrix
    # Hamming loss
    hamming_loss = np.mean(true_result != full_result)


    logits_np = logits.detach().cpu().numpy()
    

    precision_scores = []

    for i in range(n_samples):

        
        top_k = np.argsort(-logits_np[i])[:k]

        
        true_labels = np.where(true_result[i] == 1)[0]

        if len(true_labels) == 0:
            continue

        hits = len(set(top_k) & set(true_labels))

        precision_scores.append(hits / k)

    if(len(precision_scores)>1):
        precision_at_k = np.mean(precision_scores)
    else:
        precision_at_k = 0.0


    return {
    "hamming_loss": hamming_loss,
    "precision@k": precision_at_k
    }


