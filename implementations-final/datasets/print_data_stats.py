import os
import numpy as np
import scipy.io as sio

for dirname in os.listdir('./final'):
    print("dataset {0}:".format(dirname))
    print(sio.loadmat("./final/" + dirname + "/data.mat")['data'].shape)

