import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import bayesiantests as bt


# Set rope and rho values
rope=0.01
rho=1.0/10.0

comp_pairs = (('reliefseq_raw_scores.mat', 'turf_raw_scores.mat'), )
alg_names = (('ReliefSeq', 'TuRF'), )
save_names = (('reliefseq', 'turf'),)


for save_names_nxt, file_names, names in zip(save_names, comp_pairs, alg_names):

    scores_l = sio.loadmat('./raw_scores/' + file_names[0])['data']
    scores_r = sio.loadmat('./raw_scores/' + file_names[1])['data']
    scores = scores_l - scores_r
    msk = np.logical_not(np.apply_along_axis(lambda x: np.all(x == 0), 1, scores))
    scores = scores[msk, :]

    # Compute probabilities.
    pleft, prope, pright = bt.hierarchical(scores, rope, rho)
    with open('results.res', 'a') as f:
        f.write('{0}, {1}, {2}, {3}, {4}\n'.format(names[0], names[1], pleft, prope, pright))

    # Sample posterior and make simplex plot.
    samples=bt.hierarchical_MC(scores, rope, rho, names=alg_names)
    fig = bt.plot_posterior(samples, names)
    plt.savefig(save_names_nxt[0] + '_' + save_names_nxt[1] + '_final.png')

