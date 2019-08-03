import numpy as np
import os
import sys
import scipy.io as sio

PROP_ADDITIONAL_NOISE = 0.9

for dirname in os.listdir(sys.path[0] + '/final'):
    if '.' not in dirname:
        data = sio.loadmat(sys.path[0] + '/final/' + dirname + '/data.mat')['data']
        target = sio.loadmat(sys.path[0] + '/final/' + dirname + '/target.mat')['target']
        min_features_val = np.min(data)
        max_features_val = np.max(data)
        add_noise = np.random.uniform(min_features_val, max_features_val, (data.shape[0], int(np.ceil(data.shape[1]*PROP_ADDITIONAL_NOISE))))
        data = np.hstack((data, add_noise))
        data = data[:, np.random.choice(data.shape[1], data.shape[1], replace=False)]
        os.system('mkdir ' + sys.path[0] + '/noisy/' + dirname)
        sio.savemat(sys.path[0] + '/noisy/' + dirname + '/data.mat', {'data' : data})
        sio.savemat(sys.path[0] + '/noisy/' + dirname + '/target.mat', {'target' : target})
