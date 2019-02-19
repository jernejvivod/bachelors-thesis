import numpy as np
import sklearn.preprocessing


def data_discretization(X, n_bins):
    """
    This function implements the data discretization function to discrete data into n_bins

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    n_bins: {int}
        number of bins to be discretized

    Output
    ------
    X_discretized: {numpy array}, shape (n_samples, n_features)
        output discretized data, where features are digitized to n_bins
    """

    # normalize each feature
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()  # Initialize MinMaxScaler instance
    X_normalized = min_max_scaler.fit_transform(X)  # Normalize data using min max scaling

    # discretize X
    n_samples, n_features = X.shape                                  # Get number of samples and number of features.
    X_discretized = np.zeros((n_samples, n_features))                # Allocate array
    bins = np.linspace(0, 1, n_bins)                                 # Create bins using linspace to create points on unit interval.
    for i in range(n_features):                                      # Go over features.
        X_discretized[:, i] = np.digitize(X_normalized[:, i], bins)  # Add bin indices feature column

    return X_discretized  # Return matrix of bin indices.