def SURF_compute_scores(inst, attr, nan_entries, num_attributes, mcmap, NN, headers, class_type, X, y, labels_std, data_type):
    """ Unique scoring procedure for SURF algorithm. Scoring based on nearest neighbors within defined radius of current target instance. """
    scores = np.zeros(num_attributes)
    if len(NN) <= 0:
        return scores
    for feature_num in range(num_attributes):
        scores[feature_num] += compute_score(attr, mcmap, NN, feature_num, inst,
                                             nan_entries, headers, class_type, X, y, labels_std, data_type)
    return scores
