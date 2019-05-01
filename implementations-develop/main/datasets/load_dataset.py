import sys
import os
import scipy.io as sio
import pdb

def load(name, req_spec):
    """
    TODO
    """
    # Check if dataset folder exists
    if name in os.listdir(sys.path[0] + '/datasets'):
        if req_spec == 'data':  # If data requested...
            return sio.loadmat(sys.path[0] + '/datasets/' + name + '/data.mat')['data']
        if req_spec == 'target':  # If target requested...
            return sio.loadmat(sys.path[0] + '/datasets/' + name + '/target.mat')['target']
        else:  # If invalid request specifier
            raise ValueError("parameter req_spec must equal 'data' or 'target'.")
    else:  # If dataset folder does not exist, raise exception.
        raise IOError('No dataset named {0}'.format(name))
