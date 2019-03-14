def mass_based_dissimilarity(x1, x2):
    pass

def get_n_random_itrees(n, data):
    pass

def get_random_itree(data):
    """ 
    Construct and return a random itree.

    Args:
        data: data on which to build the itree
    Returns:
        an instance of an In_node class representing the root of the itree
    """

    # random_itree: define auxiliary function to implement recursion.
    def random_itree(x_in, current_height, lim):
        if current_height >= lim or data.shape[0] <= 1: # Base case check
            return Ex_node(s=data.shape[0])
        else:
            # Randomly select an attribute q.
            q = np.random.randint(data.shape[1])
            # Randomly select a split point p between min and max values of attribute q in X.
            p = np.random.uniform(np.min(data[:, q]), np.max(data[:, q]))
            # Get left and right subtrees.
            xl = data[data[:, q] < p, :]
            xr = data[data[:, q] < p, :]
            # Recursive case
            return In_node(l=random_itree(xl, current_height+1, lim),\
                           r=random_itree(xr, current_height+1, lim),\
                           split_attr=q,\
                           split_val=p)

    # Build itree
    return random_itree(data, current_depth=0, lim=100)
