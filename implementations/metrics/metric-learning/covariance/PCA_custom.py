from metric_learn import Covariance
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# Idea - reduce the dimensionality of examples and measure distances between examples in this space.

data = load_iris()['data']  # Load the Iris dataset.

# Get projected data. Use this data to measure distances between examples.
data_trans = PCA(n_components='mle').fit_transform(data)

