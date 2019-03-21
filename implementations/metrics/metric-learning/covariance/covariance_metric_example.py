from metric_learn import Covariance
from sklearn.datasets import load_iris

## Mahalanobis distance ##

# The Mahalanobis distance measures the number of standard deviations from P 
# to the mean of D. If each of these axes is re-scaled to have unit variance, 
# then the Mahalanobis distance corresponds to standard Euclidean distance in the transformed space.
# The Mahalanobis distance is thus unitless and scale-invariant, 
# and takes into account the correlations of the data set.

iris = load_iris()['data']  # Load the Iris dataset.

dists = Covariance()  # Define model
dists.fit(iris)  # Learn model
dists = dists.transform()  # Get D-dimensional learned metric space XL' in which standard Euclidean
                           # distance may be used.
data_trans = Covariance().fit_transform(iris)

# short form:
#
# data_trans = Covariance().fit_transform(iris)
# 
#
