import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle as pkl
from collections import namedtuple
import bayesiantests as bt

# Define named tuple that was used to store results.
comparePair = namedtuple('comparePair', 'algorithm1 algorithm2 scores')

# Set rope and rho values
rope=0.01
rho=1.0/10.0

# Go over stored results dictionaries in folder.
for results in glob.glob('*.p'):

    # Load next results dictionary.
    with open(results, 'rb') as f:
        results_nxt = pkl.load(f)

        # Go over pair comparisons in dictionary.
        for results_idx in results_nxt.keys():
            nxt_pair = results_nxt[results_idx]
            names = (nxt_pair.algorithm1, nxt_pair.algorithm2)
            scores = nxt_pair.scores
            msk = np.logical_not(np.apply_along_axis(lambda x: np.all(x == 0), 1, scores))
            scores = scores[msk, :]

            # Compute probabilities.
            pleft, prope, pright = bt.hierarchical(scores,rope,rho)
            with open('results.res', 'a') as f:
                f.write('{0}, {1}, {2}, {3}, {4}\n'.format(nxt_pair.algorithm1, nxt_pair.algorithm2, pleft, prope, pright))

            # Sample posterior and make simplex plot.
            samples=bt.hierarchical_MC(scores,rope,rho, names=('MultiSURF', 'MultiSURF*'))
            fig = bt.plot_posterior(samples, names)
            plt.savefig(nxt_pair.algorithm1 + '_' + nxt_pair.algorithm2 + '.png')

