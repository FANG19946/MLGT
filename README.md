In this repository we will explore Multilabel Classification using Group Testing based on the papers by Ubaru and Mazumdar.


The parts to be built as a high level overview

1. A component which can take in the Design matrix and train binary classifiers according to the rows of the matrix.
2. A component that can decode the results of group testing algorithm.
3. A component that can create the 3 types of required Design matrices
   1. Random Bernoulli (k,e) disjunct matrices
   2. Concatenated Code based matrices
   3. Bipartite Expander Graphs
4. A component to calculate our evaluation metrics
   1. Hamming Loss
   2. Precision@k

    A few more metrics that I feel we should include are 
    * Precison
    * Recall
    * F1 Score

I think this should wrap up the Paper 1
Now towards Paper 2
1. A component that can create the co-occurrence matrix of labels
2. A component that can create the SymNMF of the co-occurrence matrix *(This problem is NP Hard and will require an Iterative Machine Learning Gradient Descent approximation Algorithm)*
3. Heirarchical XML 
4. SAFFRON Decoder *(I think this part can be reused from elsewhere)*

Ok while this is all there is for these papers here are some type of ideas I have been thinking about
1. Using BERT or GLoVE Embeddings and clustering them to use group sparsity.
2. Building weighted sparse graphs and embedding them into $\mathbb{R}^p$ metric space, clustering them and imposing group sparsity
3. Finding Sparse Cuts in the label graphs and imposing group sparsity

Feel free to critique these last few ideas and add to them, if theres anything unclear do approach me. 