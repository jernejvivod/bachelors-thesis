import numpy as np
from itree_utils import It_node

def get_lowest_common_node_mass(itree, x1, x2):
    """ 
    Get mass of lowest node containing both x1 and x2.

    Args:
        itree               ... starting itree node
        x1                  ... example 1
        x2                  ... example 2
    Returns:
        mass of lowest node containing both x1 and x2
    """
    # 1. base case - if node is a leaf (no splitting value), return mass of node.
    if itree.split_val == None:
        return itree.mass

    # 2. base case - if split attribute value above split value for one and below split value
    # for other examples, return mass of node.
    if (x1[itree.split_attr] < itree.split_val) != (x2[itree.split_attr] < itree.split_val):
        return itree.mass

    # 1. Recursive case - both split attribute value < split value:
    # Go to left subtree.
    if x1[itree.split_attr] < itree.split_val and x2[itree.split_attr] < itree.split_val:
        return get_lowest_common_node_mass(itree.l, x1, x2)


    # 2. Recursive case - both split attribute value > split value:
    # Go to right subtree.
    if x1[itree.split_attr] >= itree.split_val and x2[itree.split_attr] >= itree.split_val:
        return get_lowest_common_node_mass(itree.r, x1, x2)


def mass_based_dissimilarity(x1, x2, random_itrees, example_subset_size):
    """ 
    Get mass based dissimilarity of examples x1 and x2.

    Args:
        x1                  ... example 1
        x2                  ... example 2
        random_itrees       ... list of random itrees built on training examples
        example_subset_size ... size of training example subset used to build random itrees
    Returns:
        mass based dissimilarity of examples x1 and x2
    """
    # In each i-tree, find lowest nodes containing both x and y
    # TODO: parallelize
    sum_masses = 0
    for itree in random_itrees:
        # Divide each sum by size of subset used to construct the trees.
        sum_masses += get_lowest_common_node_mass(itree, x1, x2)/example_subset_size
    return (1/len(random_itrees)) * sum_masses  # Divide by number of space partitioning models.


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
            xr = x_in[x_in[:, q] >= p, :]
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
        # Get a random sample of training examples to build next random itree.
        data_sub = data[np.random.choice(data.shape[0], sub_size, replace=False), :]
        random_trees[k] = get_random_itree(data_sub)  # Get next random itree 
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

    # traverse: traverse itree with example and increment masses of visited nodes
    def traverse(example, it_node):
        # base case - in leaf
        if it_node.l == None and it_node.r == None:
            it_node.mass += 1
        # if split attribute value lower than split value
        elif example[it_node.split_attr] < it_node.split_val:
            it_node.mass += 1
            traverse(example, it_node.l)  # Traverse left subtree.
        # if split attribute value greater or equal to split value
        else:
            it_node.mass += 1
            traverse(example, it_node.r)  # Traverse right subtree.

    # compute_masses: compute masses of nodes in itree
    def compute_masses(data, itree):
        for example in data:
            traverse(example, itree)

    # TODO: parallelize!
    for itree in random_itrees:  # Go over itrees and set masses of nodes.
        compute_masses(data, itree)


# If running as main script, perform simple test.
if __name__ == "__main__":
    import numpy as np
    import scipy.io as sio
    import pdb
    from itree_utils import It_node
    examples = sio.loadmat('./data/examples.mat')['examples']
    random_itrees = get_n_random_itrees(10, examples, examples.shape[0])
    get_node_masses(examples, random_itrees)
    me = mass_based_dissimilarity(examples[0, :], examples[1, :] , random_itrees, examples.shape[0])
    print(me)
