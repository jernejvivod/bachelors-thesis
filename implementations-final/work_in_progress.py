import numpy as np
from sklearn.model_selection import KFold, RepeatedKFold, StratifiedKFold, train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import scipy.io as sio

import pdb

from algorithms.relieff import Relieff
from algorithms.iterative_relief import IterativeRelief
from algorithms.irelief import IRelief

from algorithms.augmentations import lmnn
from algorithms.augmentations import nca
from algorithms.augmentations import itml

"""

##### comparisons I: ####
ReliefF - k-nearest     *
ReliefF -- diff         *
ReliefF -- eksp. rang   *
#########################

##### comparisons II: ###
ReliefF - LMNN (IN)
ReliefF - LMNN (OUT)
#########################

##### comparisons III: ##
ReliefF - NCA (IN)
ReliefF - NCA (OUT)
#########################

##### comparisons IV: ###
ReliefF - ITML (IN)
ReliefF - ITML (OUT)
#########################

"""

def run(name, func):

    data = sio.loadmat('datasets/ecoli/data.mat')['data']
    target = np.ravel(sio.loadmat('datasets/ecoli/target.mat')['target'])

    # Number of folds and runs in cross validation process.
    NUM_SPLITS = 10
    NUM_REPEATS = 10

    # k parameter value.
    PARAM_K = 10

    # Initialize classifier.
    clf = KNeighborsClassifier(n_neighbors=5)

    # Initialize RBA.
    rba = IterativeRelief()

    # Allocate vector for storing accuracies for each fold
    # and fold counter that indexes it.
    cv_results = np.empty(NUM_SPLITS * NUM_REPEATS, dtype=np.float)
    idx_fold = 0

    # Initialize CV iterator.
    kf = RepeatedKFold(n_splits=NUM_SPLITS, n_repeats=NUM_REPEATS, random_state=1)

    # Initialize cross validation strategy.
    cv_strategy = KFold(n_splits=3, random_state=1)

    # Go over folds.
    for train_idx, test_idx in kf.split(data):
        
        # Split data into training set, validation set and test set
        data_train = data[train_idx]
        target_train = target[train_idx]
        data_test = data[test_idx]
        target_test = target[test_idx]
        data_train, data_val, target_train, target_val = train_test_split(data_train, target_train, test_size=0.3, random_state=1)

        # training set -- data_train, target_train
        # validation set -- data_val, target_val
        # test set -- data_test, target_test

        # Perform grid search on validation set.
        clf_pipeline = Pipeline([('scaling', StandardScaler()), ('rba', rba), ('clf', clf)])
        param_grid = {
            'rba__n_features_to_select': np.arange(1, 50)
        }
        gs = GridSearchCV(clf_pipeline, param_grid=param_grid, cv=cv_strategy, verbose=True, iid=False, n_jobs=-1)
        gs.fit(data_val, target_val)

        # Train model on training set.
        trained_model = gs.best_estimator_.fit(data_train, target_train)

        # Compute classification accuracy on test set and store in results vector
        res = trained_model.predict(data_test)
        fold_score = accuracy_score(target_test, res)
        cv_results[idx_fold] = fold_score
        idx_fold += 1

        print("finished fold {0}/{1}".format(idx_fold-1, NUM_SPLITS*NUM_REPEATS))

    sio.savemat(name + '.mat', {'res' : cv_results})

# run('nca_final', nca.get_dist_func)
# run('itml_final', itml.get_dist_func)
run('iterative_relief', None)


