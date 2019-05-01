def SURFstar_compute_scores(inst, attr, nan_entries, num_attributes, mcmap, NN_near, NN_far, headers, class_type, X, y, labels_std, data_type):
    """ Unique scoring procedure for SURFstar algorithm. Scoring based on nearest neighbors within defined radius, as well as
    'anti-scoring' of far instances outside of radius of current target instance"""
    scores = np.zeros(num_attributes)  # Allocate array for storing the scores.
    for feature_num in range(num_attributes):  # Go over features.
        if len(NN_near) > 0:  # If any near neighbours.
            scores[feature_num] += compute_score(attr, mcmap, NN_near, feature_num, inst,
                                                 nan_entries, headers, class_type, X, y, labels_std, data_type)
        # Note that we are using the near scoring loop in 'compute_score' and then just subtracting it here, in line with original SURF* paper.
        if len(NN_far) > 0:  # If any far neighbours.
            scores[feature_num] -= compute_score(attr, mcmap, NN_far, feature_num, inst,
                                                 nan_entries, headers, class_type, X, y, labels_std, data_type)
    return scores
