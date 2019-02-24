import numpy as np
import pdb

def movsum(l, k):
	final_arr = np.zeros((1, len(l)))
	for m in np.arange(k):
		final_arr = final_arr + np.roll(l, -m)
	return final_arr
