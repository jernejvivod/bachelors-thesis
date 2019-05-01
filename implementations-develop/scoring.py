def compute_score(attr, mcmap, NN, feature, inst, nan_entries, headers, class_type, X, y, labels_std, data_type, near=True):
    """Flexible feature scoring method that can be used with any core Relief-based method. Scoring proceeds differently
    based on whether endpoint is binary, multiclass, or continuous. This method is called for a single target instance
    + feature combination and runs over all items in NN. """

    fname = headers[feature]  # feature identifier
    ftype = attr[fname][0]  # feature type
    ctype = class_type  # class type (binary, multiclass, continuous)
    diff_hit = diff_miss = 0.0  # Tracks the score contribution
    # Tracks the number of hits/misses. Used in normalizing scores by 'k' in ReliefF, and by m or h in SURF, SURF*, MultiSURF*, and MultiSURF
    count_hit = count_miss = 0.0
    # Initialize 'diff' (The score contribution for this target instance and feature over all NN)
    diff = 0
    # mmdiff = attr[fname][3] # Max/Min range of values for target feature

    datalen = float(len(X))

    # If target instance is missing, then a 'neutral' score contribution of 0 is returned immediately since all NN comparisons will be against this missing value.
    if nan_entries[inst][feature]:
        return 0.
    # Note missing data normalization below regarding missing NN feature values is accomplished by counting hits and misses (missing values are not counted) (happens in parallel with hit/miss imbalance normalization)

    xinstfeature = X[inst][feature]  # value of target instances target feature.

    #--------------------------------------------------------------------------
    if ctype == 'binary':
        for i in range(len(NN)):
            if nan_entries[NN[i]][feature]:  # skip any NN with a missing value for this feature.
                continue

            xNNifeature = X[NN[i]][feature]

            if near:  # SCORING FOR NEAR INSTANCES
                if y[inst] == y[NN[i]]:   # HIT
                    count_hit += 1
                    if ftype == 'continuous':
                        # diff_hit -= abs(xinstfeature - xNNifeature) / mmdiff #Normalize absolute value of feature value difference by max-min value range for feature (so score update lies between 0 and 1)
                        diff_hit -= ramp_function(data_type, attr, fname, xinstfeature, xNNifeature)
                    else:  # discrete feature
                        if xinstfeature != xNNifeature:  # A difference in feature value is observed
                            # Feature score is reduced when we observe feature difference between 'near' instances with the same class.
                            diff_hit -= 1
                else:  # MISS
                    count_miss += 1
                    if ftype == 'continuous':
                        #diff_miss += abs(xinstfeature - xNNifeature) / mmdiff
                        diff_miss += ramp_function(data_type, attr, fname,
                                                   xinstfeature, xNNifeature)
                    else:  # discrete feature
                        if xinstfeature != xNNifeature:  # A difference in feature value is observed
                            # Feature score is increase when we observe feature difference between 'near' instances with different class values.
                            diff_miss += 1

            else:  # SCORING FOR FAR INSTANCES (ONLY USED BY MULTISURF* BASED ON HOW CODED)
                if y[inst] == y[NN[i]]:   # HIT
                    count_hit += 1
                    if ftype == 'continuous':

                        #diff_hit -= abs(xinstfeature - xNNifeature) / mmdiff  #Hits differently add continuous value differences rather than subtract them 
                        diff_hit -= (1-ramp_function(data_type, attr, fname, xinstfeature, xNNifeature)) #Sameness should yield most negative score
                    else: #discrete feature
                        if xinstfeature == xNNifeature: # The same feature value is observed (Used for more efficient 'far' scoring, since there should be fewer same values for 'far' instances)
                            diff_hit -= 1 # Feature score is reduced when we observe the same feature value between 'far' instances with the same class.
                else:  # MISS
                    count_miss += 1
                    if ftype == 'continuous':
                        #diff_miss += abs(xinstfeature - xNNifeature) / mmdiff #Misses differntly subtract continuous value differences rather than add them 
                        diff_miss += (1-ramp_function(data_type, attr, fname, xinstfeature, xNNifeature)) #Sameness should yield most negative score
                    else: #discrete feature
                        if xinstfeature == xNNifeature: # The same feature value is observed (Used for more efficient 'far' scoring, since there should be fewer same values for 'far' instances)
                            diff_miss += 1 # Feature score is increased when we observe the same feature value between 'far' instances with different class values.

        """ Score Normalizations:
        *'n' normalization dividing by the number of training instances (this helps ensure that all final scores end up in the -1 to 1 range
        *'k','h','m' normalization dividing by the respective number of hits and misses in NN (after ignoring missing values), also helps account for class imbalance within nearest neighbor radius)"""
        if count_hit == 0.0 or count_miss == 0.0:  # Special case, avoid division error
            if count_hit == 0.0 and count_miss == 0.0:
                return 0.0
            elif count_hit == 0.0:
                diff = (diff_miss / count_miss) / datalen
            else:  # count_miss == 0.0
                diff = (diff_hit / count_hit) / datalen
        else:  # Normal diff normalization
            diff = ((diff_hit / count_hit) + (diff_miss / count_miss)) / datalen

    #--------------------------------------------------------------------------
    elif ctype == 'multiclass':
        class_store = dict() #only 'miss' classes will be stored
        #missClassPSum = 0

        for each in mcmap:
            if(each != y[inst]):  # Identify miss classes for current target instance.
                class_store[each] = [0, 0]
                #missClassPSum += mcmap[each]

        for i in range(len(NN)):
            if nan_entries[NN[i]][feature]:  # skip any NN with a missing value for this feature.
                continue

            xNNifeature = X[NN[i]][feature]

            if near:  # SCORING FOR NEAR INSTANCES
                if(y[inst] == y[NN[i]]):  # HIT
                    count_hit += 1
                    if ftype == 'continuous':
                        #diff_hit -= abs(xinstfeature - xNNifeature) / mmdiff
                        diff_hit -= ramp_function(data_type, attr, fname, xinstfeature, xNNifeature) 
                    else: #discrete feature
                        if xinstfeature != xNNifeature:
                            # Feature score is reduced when we observe feature difference between 'near' instances with the same class.
                            diff_hit -= 1
                else:  # MISS
                    for missClass in class_store:
                        if(y[NN[i]] == missClass):  # Identify which miss class is present
                            class_store[missClass][0] += 1
                            if ftype == 'continuous':
                                #class_store[missClass][1] += abs(xinstfeature - xNNifeature) / mmdiff
                                class_store[missClass][1] += ramp_function(
                                    data_type, attr, fname, xinstfeature, xNNifeature)
                            else:  # discrete feature
                                if xinstfeature != xNNifeature:
                                    # Feature score is increase when we observe feature difference between 'near' instances with different class values.
                                    class_store[missClass][1] += 1

            else:  # SCORING FOR FAR INSTANCES (ONLY USED BY MULTISURF* BASED ON HOW CODED)
                if(y[inst] == y[NN[i]]):  # HIT
                    count_hit += 1
                    if ftype == 'continuous':
                        #diff_hit -= abs(xinstfeature - xNNifeature) / mmdiff  #Hits differently add continuous value differences rather than subtract them 
                        diff_hit -= (1-ramp_function(data_type, attr, fname, xinstfeature, xNNifeature)) #Sameness should yield most negative score
                    else: #discrete features
                        if xinstfeature == xNNifeature:
                            # Feature score is reduced when we observe the same feature value between 'far' instances with the same class.
                            diff_hit -= 1
                else:  # MISS
                    for missClass in class_store:
                        if(y[NN[i]] == missClass):
                            class_store[missClass][0] += 1
                            if ftype == 'continuous':
                                #class_store[missClass][1] += abs(xinstfeature - xNNifeature) / mmdiff
                                class_store[missClass][1] += (1-ramp_function(data_type, attr, fname, xinstfeature, xNNifeature)) #Sameness should yield most negative score
                            else: #discrete feature
                                if xinstfeature == xNNifeature:
                                    # Feature score is increased when we observe the same feature value between 'far' instances with different class values.
                                    class_store[missClass][1] += 1

        """ Score Normalizations:
        *'n' normalization dividing by the number of training instances (this helps ensure that all final scores end up in the -1 to 1 range
        *'k','h','m' normalization dividing by the respective number of hits and misses in NN (after ignoring missing values), also helps account for class imbalance within nearest neighbor radius)
        * multiclass normalization - accounts for scoring by multiple miss class, so miss scores don't have too much weight in contrast with hit scoring. If a given miss class isn't included in NN
        then this normalization will account for that possibility. """
        # Miss component
        for each in class_store:
            count_miss += class_store[each][0]

        if count_hit == 0.0 and count_miss == 0.0:
            return 0.0
        else:
            if count_miss == 0:
                pass
            else: #Normal diff normalization
                for each in class_store: #multiclass normalization
                    diff += class_store[each][1] * (class_store[each][0] / count_miss) * len(class_store)# Contribution of given miss class weighted by it's observed frequency within NN set.
                diff = diff / count_miss #'m' normalization
            
            #Hit component: with 'h' normalization
            if count_hit == 0:
                pass
            else:
                diff += (diff_hit / count_hit)

        diff = diff / datalen  # 'n' normalization

    #--------------------------------------------------------------------------
    else:  # CONTINUOUS endpoint
        same_class_bound = labels_std

        for i in range(len(NN)):
            if nan_entries[NN[i]][feature]:  # skip any NN with a missing value for this feature.
                continue

            xNNifeature = X[NN[i]][feature]

            if near:  # SCORING FOR NEAR INSTANCES
                if abs(y[inst] - y[NN[i]]) < same_class_bound:  # HIT approximation
                    count_hit += 1
                    if ftype == 'continuous':
                        #diff_hit -= abs(xinstfeature - xNNifeature) / mmdiff
                        diff_hit -= ramp_function(data_type, attr, fname, xinstfeature, xNNifeature)
                    else:  # discrete feature
                        if xinstfeature != xNNifeature:
                            # Feature score is reduced when we observe feature difference between 'near' instances with the same 'class'.
                            diff_hit -= 1
                else:  # MISS approximation
                    count_miss += 1
                    if ftype == 'continuous':
                        #diff_miss += abs(xinstfeature - xNNifeature) / mmdiff
                        diff_miss += ramp_function(data_type, attr, fname,
                                                   xinstfeature, xNNifeature)
                    else:  # discrete feature
                        if xinstfeature != xNNifeature:
                            # Feature score is increase when we observe feature difference between 'near' instances with different class value.
                            diff_miss += 1

            else:  # SCORING FOR FAR INSTANCES (ONLY USED BY MULTISURF* BASED ON HOW CODED)
                if abs(y[inst] - y[NN[i]]) < same_class_bound:  # HIT approximation
                    count_hit += 1
                    if ftype == 'continuous':
                        #diff_hit += abs(xinstfeature - xNNifeature) / mmdiff
                        diff_hit -= (1-ramp_function(data_type, attr, fname, xinstfeature, xNNifeature)) #Sameness should yield most negative score
                    else: #discrete feature
                        if xinstfeature == xNNifeature:
                            # Feature score is reduced when we observe the same feature value between 'far' instances with the same class.
                            diff_hit -= 1
                else:  # MISS approximation
                    count_miss += 1
                    if ftype == 'continuous':
                        #diff_miss -= abs(xinstfeature - xNNifeature) / mmdiff
                        diff_miss += (1-ramp_function(data_type, attr, fname, xinstfeature, xNNifeature)) #Sameness should yield most negative score
                    else: #discrete feature
                        if xinstfeature == xNNifeature:
                            # Feature score is increased when we observe the same feature value between 'far' instances with different class values.
                            diff_miss += 1

        """ Score Normalizations:
        *'n' normalization dividing by the number of training instances (this helps ensure that all final scores end up in the -1 to 1 range
        *'k','h','m' normalization dividing by the respective number of hits and misses in NN (after ignoring missing values), also helps account for class imbalance within nearest neighbor radius)"""

        if count_hit == 0.0 or count_miss == 0.0:  # Special case, avoid division error
            if count_hit == 0.0 and count_miss == 0.0:
                return 0.0
            elif count_hit == 0.0:
                diff = (diff_miss / count_miss) / datalen
            else:  # count_miss == 0.0
                diff = (diff_hit / count_hit) / datalen
        else:  # Normal diff normalization
            diff = ((diff_hit / count_hit) + (diff_miss / count_miss)) / datalen

    return diff
