import numpy as np
import itertools
import pdb

sample_data = np.array([[1, 2, 3, 1, 2, 3], [122, 22, 3, 55, 811, 122], [1, 8, 9, 8, 9, 10]])

def mp_dissim(x1, x2, p, data):

    def get_region_data_mass(data, interval, dimension):
        return np.sum(np.logical_and(data[:, dimension] >= interval[0],\
                                     data[:, dimension] <= interval[1]))

    def get_region(x1i, x2i, lam):
        return np.array([np.min((x1i, x2i)) - lam, np.max((x1i, x2i)) + lam])

    def get_lam(data, dimension):
        return np.std(data[:, dimension]) / 2.0

    res = 0.0
    for dimension, (x1i, x2i) in enumerate(zip(x1, x2)):
        lam = get_lam(data, dimension)
        interval = get_region(x1i, x2i, lam)
        res += (get_region_data_mass(data, interval, dimension)/data.shape[0])**p
    
    return res**(1/p)

if __name__ == "__main__":
   diss = mp_dissim(sample_data[0, :], sample_data[1, :], 1, sample_data)
   print(diss)
