import numpy as np
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import rankdata


class RandomSelector(BaseEstimator, TransformerMixin):

    def __init__(self, n_features_to_select):
        self.n_features_to_select = n_features_to_select

    def _rand_sel(data, **kwargs):
        """
        Assign random weights to features and rank them

        Args:
            data : Array[np.float64] -- training examples
            kwargs -- ignored

        Returns:
            random ranking of features and randomly assigned feature weights
        """
        weights = np.random.rand(data.shape[1])
        rank = rankdata(-weights, method='ordinal')
        return rank, weights

    def fit(self, data, target):
        self.rank, self.weights = _rand_sel(data)
        return self

    def transform(self, data):
        msk = self.rank <= self.n_features_to_select  # Compute mask.
        return data[:, msk]  # Perform feature selection.

    def fit_transform(self, data, target):
        self.fit(data, target)
        return self.transform(data)

