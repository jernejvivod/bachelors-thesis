def MultiSURFstar_compute_scores(inst, attr, nan_entries, num_attributes, mcmap, NN_near, NN_far, headers, class_type, X, y, labels_std, data_type):
    """ Unique scoring procedure for MultiSURFstar algorithm. Scoring based on 'extreme' nearest neighbors within defined radius, as
    well as 'anti-scoring' of extreme far instances defined by outer radius of current target instance. """
    scores = np.zeros(num_attributes)

    for feature_num in range(num_attributes):
        if len(NN_near) > 0:
            scores[feature_num] += compute_score(attr, mcmap, NN_near, feature_num, inst,
                                                 nan_entries, headers, class_type, X, y, labels_std, data_type)
        # Note that we add this term because we used the far scoring above by setting 'near' to False.  This is in line with original MultiSURF* paper.
        if len(NN_far) > 0:
            scores[feature_num] += compute_score(attr, mcmap, NN_far, feature_num, inst,
                                                 nan_entries, headers, class_type, X, y, labels_std, data_type, near=False)

    return scores
