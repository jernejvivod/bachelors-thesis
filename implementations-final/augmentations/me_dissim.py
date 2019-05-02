import numpy as np
from functools import partial

import pdb

class It_node:
    def __init__(self, l, r, split_attr, split_val, level, mass_comp=0):
        self.l = l                    # left subtree
        self.r = r                    # right subree
        self.split_attr = split_attr  # split attribute/dimension
        self.split_val = split_val    # split value
        self.level = level            # node level
        self.mass = 0                 # node mass (number of examples in region)

    # to_string: return a string encoding some information about the node
    def to_string(self):
        return "split_attr={0}, split_val={1}, level={2}, mass={3}".format(self.split_attr,\
                                                                 self.split_val,\
                                                                 self.level,
                                                                 self.mass)

class MeDissimilarity:

    def __init__(self, data):
        self.data = data

    def get_lowest_common_node_mass(self, itree, x1, x2):
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
            return self.get_lowest_common_node_mass(itree.l, x1, x2)


        # 2. Recursive case - both split attribute value > split value:
        # Go to right subtree.
        if x1[itree.split_attr] >= itree.split_val and x2[itree.split_attr] >= itree.split_val:
            return self.get_lowest_common_node_mass(itree.r, x1, x2)


    def mass_based_dissimilarity(self, x1, x2):
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
        for itree in self.random_itrees:
            # Divide each sum by size of subset used to construct the trees.
            sum_masses += self.get_lowest_common_node_mass(itree, x1, x2)/self.subs_size
        return (1/len(self.random_itrees)) * sum_masses  # Divide by number of space partitioning models.


    def get_random_itree(self, data_sub):
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


    def get_n_random_itrees(self, n, subs_size):
        """ 
        Construct and return n random itrees in a list.

        Args:
            n: number of random itrees to construct
            data: data on which to build the itree
            sub_size: size of subset of examples used to build the tree
        Returns:
            n random itrees in a list
        """
        random_itrees = np.empty(n, dtype=object)  # Allocate list for storing the trees.
        # TODO: parallelize!
        for k in np.arange(n):
            # Get a random sample of training examples to build next random itree.
            data_sub = self.data[np.random.choice(self.data.shape[0], subs_size, replace=False), :]
            random_itrees[k] = self.get_random_itree(data_sub)  # Get next random itree 
        self.random_itrees = random_itrees
        self.subs_size = subs_size


    def get_node_masses(self):
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
        def compute_masses(itree):
            for example in self.data:
                traverse(example, itree)

        # TODO: parallelize!
        for itree in self.random_itrees:  # Go over itrees and set masses of nodes.
            compute_masses(itree)

    def get_dissim_func(self, num_itrees):
        """
        get function that takes two examples and computes their mass based dissimilarity

        Args:
            data : Array[np.float64] -- training examples
            num_itrees : np.int -- number of i-trees to use
            mass_based_dissimilarity : Callable[[Array[np.float64], Array[np.float64], Array[object], np.int], np.float64] -- 
                function that implements mass based dissimilarity on learned i-trees.
        Returns:
            Callable[[Array[np.float64], Array[np.float64]], np.float64] -- learned dissimilarity function
        """
        self.get_n_random_itrees(num_itrees, self.data.shape[0])
        self.get_node_masses()

        def dissim_func(x1, x2):
            if x1.ndim == 1 and x2.ndim == 1:
                return self.mass_based_dissimilarity(x1, x2)
            elif x1.ndim == 1 and x2.ndim == 2:
                dissim_part = partial(self.mass_based_dissimilarity, x1)
                return np.apply_along_axis(dissim_part, 1, x2)
            elif x1.ndim == 2 and x2.ndim == 1:
                dissim_part = partial(self.mass_based_dissimilarity, x2)
                return np.apply_along_axis(self.mass_based_dissimilarity, 1, x1)
            elif x1.ndim == 2 and x2.ndim == 2:
                res = np.array(x1.shape[0], dtype=float)
                for idx, (r1, r2) in enumerate(zip(x1, x2)):
                    res[idx] = self.mass_based_dissimilarity(r1, r2)

        return dissim_func

# If running as main script, perform simple test.
if __name__ == "__main__":
    import numpy as np
    import scipy.io as sio
    import pdb
    examples = sio.loadmat('./data/examples.mat')['examples']
    me = MeDissimilarity(examples)
    dissim_func = me.get_dissim_func(10)
    res = dissim_func(examples[0, :], examples[0, :])
    print(res)
