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
    z = A.T @ y
    return z

def dataset_params(trainset, e_ratio):
    trainset = np.array(trainset,dtype=bool)
    n_tests, n_labels = trainset.shape

    label_sparsity = trainset.sum(axis=1)
    label_frequency = trainset.sum(axis=0)
    mean_frequency = np.mean(label_frequency)
    k=int(np.median(label_sparsity))
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
        Y_test[:, i] = torch.any(label_matrix[:, label_mask], dim=1)

    
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

