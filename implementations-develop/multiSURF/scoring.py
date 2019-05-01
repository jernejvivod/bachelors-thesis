def MultiSURF_compute_scores(inst, attr, nan_entries, num_attributes, mcmap, NN_near, headers, class_type, X, y, labels_std, data_type):
    """ Unique scoring procedure for MultiSURF algorithm. Scoring based on 'extreme' nearest neighbors within defined radius of current target instance. """
    scores = np.zeros(num_attributes)
    for feature_num in range(num_attributes):
        if len(NN_near) > 0:
            scores[feature_num] += compute_score(attr, mcmap, NN_near, feature_num, inst,
                                                 nan_entries, headers, class_type, X, y, labels_std, data_type)

    return scores
