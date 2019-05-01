import numpy as np
import itertools


def mp_dissim(x1, x2, p, data):
    """ mp_dissim: compute mp dissimilarity of examples x1 and x2

    --- parameters ---

    x1 ... first example
    x2 ... second exapmle
    p  ... the p constant of the norm equation
    data ... data matrix

    ------

    returns:
        dissimilarity value of the two examples

    """


    # Define accumulator for the sum.
    res = 0.0

    # Go over dimensions in the two examples.
    for dimension, (x1i, x2i) in enumerate(zip(x1, x2)):
        lam = np.std(data[:, dimension]) / 2.0  # Compute lambda.
        # Compute region for next pair of values in this dimension.
        interval = np.array([np.min((x1i, x2i)) - lam, np.max((x1i, x2i)) + lam])
        region_data_mass = np.sum(np.logical_and(data[:, dimension] >= interval[0],\
                                                 data[:, dimension] <= interval[1]))
        res += (region_data_mass/data.shape[0])**p  # Compute next term in sum.
    
    return res**(1/p)  # Return result.

if __name__ == "__main__":

    # Define sample data matrix.
    sample_data = np.array([[1, 2, 3, 1, 2, 3], [122, 22, 3, 55, 811, 122], [1, 8, 9, 8, 9, 10]])

    # compute dissimilartiy and print results.
    diss = mp_dissim(sample_data[0, :], sample_data[1, :], 1, sample_data)
    print(diss)
