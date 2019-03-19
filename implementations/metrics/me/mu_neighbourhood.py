import sklearn.metrics

def get_mu_neighbourhood_masses(examples, mu, me_func):
    """
    get mu-neighbourhood masses of training examples

    Args:
        examples ... matrix of training examples
        mu       ... the mu parameter (me-dissimilarity threshold)
        me_func  ... initialized me-dissimilarity function that accepts just two examples
                     and computes the dissimilarity value.
    Returns:
        array of mu-neighbourhood masses for each example in examples matrix
        (the indices correspond)
    """
    
    # Initialize generator of pairwise dissimilarities between examples.
    gen = sklearn.metrics.pairwise_distances_chunked(\
            examples, metric=me_func, n_jobs=-1, working_memory=0)
    # Allocate array for storing results (mu-neighbourhood masses).
    mu_neigh_masses = np.empty(examples.shape[0], dtype=float)
    for k in np.arange(examples.shape[0]):  # Go over example indices.
        dists_nxt = next(gen)[0]            # Get pairwise dissimilarities for next example.
        dists_nxt[k] = np.inf               # Set self-dissimmilarity to inf.
        mu_neigh_masses[k] = np.sum(dists_nxt <= mu)  # Save result.
    return mu_neigh_masses

# If running as main script, make short test.
if __name__ == '__main__':
    import numpy as np
    import scipy.io as sio
    import pdb
    from itree_utils import It_node
    from me import get_n_random_itrees
    from me import get_node_masses
    from me import mass_based_dissimilarity

    examples = sio.loadmat('./data/examples.mat')['examples']
    sub_size = examples.shape[0]
    random_itrees = get_n_random_itrees(10, examples, examples.shape[0])
    get_node_masses(examples, random_itrees)

    # Note contruction of me_func.
    res = get_mu_neighbourhood_masses(\
            examples, 0.2, lambda x1, x2: mass_based_dissimilarity(x1, x2, random_itrees, sub_size))

