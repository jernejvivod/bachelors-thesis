import numpy as np
from scipy.stats import norm

from algorithms.relief import Relief
from algorithms.relieff import Relieff
from algorithms.reliefseq import ReliefSeq
from algorithms.reliefmss import ReliefMSS
from algorithms.turf import TuRF

import scipy.io as sio

NUM_NOISE_LIM = 10
NUM_SAMPLES = 100
K_PARAM = 10

rbas = {'Relief' : Relief(), 
        'ReliefF' : Relieff(k=K_PARAM), 
        'ReliefMSS' : ReliefMSS(k=K_PARAM), 
        'ReliefSeq' : ReliefSeq(k_max = 3), 
        'TuRF' : TuRF()}

res = dict.fromkeys(rbas.keys(), np.empty((2, NUM_NOISE_LIM+1), dtype=np.float))

for alg_name in rbas.keys():

    print("Testing {0}.".format(alg_name))

    rba = rbas[alg_name]

    for num_noise in np.arange(NUM_NOISE_LIM+1):

        print("{0}/{1}".format(num_noise, NUM_NOISE_LIM))

        feature_inf1 = np.linspace(0, 1, NUM_SAMPLES)
        feature_inf2 = 300*norm.pdf(np.arange(NUM_SAMPLES), np.int(NUM_SAMPLES/2), np.int(NUM_SAMPLES/10))

        target1 = (feature_inf1 > 0.5).astype(np.int)
        target2 = np.logical_or(feature_inf1 < 0.1, feature_inf1 > 0.6).astype(np.int)

        noise = np.random.normal(0.5, 1, (NUM_SAMPLES, num_noise))

        data1 = np.hstack((feature_inf1[np.newaxis].T, noise))
        data2 = np.hstack((feature_inf2[np.newaxis].T, noise))

        rba.fit(data1, target1)
        (res[alg_name])[0, num_noise] = rba.weights[0]

        rba.fit(data2, target2)
        (res[alg_name])[1, num_noise] = rba.weights[0]

for res_key in res.keys():
    sio.savemat('./monotonic-nonmonotonic-results/' + res_key + '_m.mat', {'data' : (res[res_key])[0, :]})
    sio.savemat('./monotonic-nonmonotonic-results/' + res_key + '_nm.mat', {'data' : (res[res_key])[1, :]})




