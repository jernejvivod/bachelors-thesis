import scipy.io as sio

from julia import Julia
jl = Julia(compiled_modules=False)

import os

script_path = os.path.abspath(__file__)
get_dist_func = jl.include(script_path[:script_path.rfind('/')] + "/me_dissim2.jl")
data = sio.loadmat('data.mat')['data']

dist_func = get_dist_func(10, data)

