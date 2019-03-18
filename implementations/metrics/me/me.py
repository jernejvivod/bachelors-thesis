import numpy as np





def mass_based_dissimilarity(x1, x2):
    pass










def get_random_itree(data_sub):
    """ 
    Construct and return a random itree.

    Args:
        data_sub: subset of training examples on which to build the itree
    Returns:
        an instance of an In_node class representing the root of the itree
    """

    # random_itree: define auxiliary function to implement recursion.
    def random_itree(x_in, current_height, lim):
        if current_height >= lim or x_in.shape[0] <= 1: # Base case check
            return It_node(l=None, r=None, split_attr=None, split_val=None, level=current_height)
        else:
            # Randomly select an attribute q.
            q = np.random.randint(x_in.shape[1])
            # Randomly select a split point p between min and max values of attribute q in X.
            p = np.random.uniform(np.min(x_in[:, q]), np.max(x_in[:, q]))
            # Get left and right subtrees.
            xl = x_in[x_in[:, q] < p, :]
            xr = x_in[x_in[:, q] > p, :]
            # Recursive case
            return It_node(l=random_itree(xl, current_height+1, lim),\
                           r=random_itree(xr, current_height+1, lim),\
                           split_attr=q,\
                           split_val=p,\
                           level=current_height)

    # Build itree
    return random_itree(data_sub, current_height=0, lim=10)










def get_n_random_itrees(n, data, sub_size):
    """ 
    Construct and return n random itrees in a list.

    Args:
        n: number of random itrees to construct
        data: data on which to build the itree
        sub_size: size of subset of examples used to build the tree
    Returns:
        n random itrees in a list
    """
    random_trees = np.empty(n, dtype=object)  # Allocate list for storing the trees.
    # TODO: parallelize!
    for k in np.arange(n):
        data_sub = data[np.random.choice(data.shape[0], sub_size, replace=False), :]
        random_trees[k] = get_random_itree(data_sub) 
    return random_trees


def get_node_masses(data, random_itrees):
    """ 
    Set mass property of each node by traversing trees with each instance
    in training set.

    Args:
        data: training data
        random_itrees: list of random itrees to traverse and set mass property
    Returns:
        list of random itrees with set mass property
    """

    def traverse(example, it_node):
        if it_node.l == None and it_node.r == None:
            it_node.mass += 1
        elif example[it_node.split_attr] < it_node.split_val:
            it_node.mass += 1
            traverse(example, it_node.l)
        else:
            it_node.mass += 1
            traverse(example, it_node.r)


    def compute_masses(data, itree):
        for example in data:
            traverse(example, itree)

    # TODO: parallelize!
    for itree in random_itrees:
        compute_masses(data, itree)



if __name__ == "__main__":
    import numpy as np
    import pdb
    from itree_utils import It_node

    data = np.array([[1, 2, 3], [3, 2, 1], [6, 7 ,2]])
    random_itrees = get_n_random_itrees(10, data, 1)
    get_node_masses(data, random_itrees)
