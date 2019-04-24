import numpy as np
from sklearn.pipeline import Pipeline
from relief import Relief
from augmentations import covariance, me_dissim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold

import scipy.io as sio


### LOADING DATASET ######

data = sio.loadmat('datasets/basic-artificial-1/data.mat')['data']
target = np.ravel(sio.loadmat('datasets/basic-artificial-1/target.mat')['target'])

##########################


### METRIC FUNTIONS ############

manhattan = lambda x1, x2 : np.sum(np.abs(x1-x2), 1)
manhattan.kind = "Manhattan"

euclidean = lambda x1, x2 : np.sum(np.abs(x1-x2)**2, 1)**(1.0/2.0)
euclidean.kind = 'Euclidean'

################################


### LEARNED METRIC FUNCTIONS ###

# covariance/Mahalanobis distance
covariance_dist_func = covariance.get_dist_func(data, target)
covariance_dist_func.kind = "Mahalanobis distance"

# Mass based dissimilarity
NUM_ITREES = 10
me_dissim_dist_func = lambda _, i1, i2: me_dissim.MeDissimilarity(data).get_dissim_func(NUM_ITREES)(data[i1, :], data[i2, :])
me_dissim_dist_func.kind = "Mass based dissimilarity"

#################################




### INITIAL PARAMETERS ###

n_features_to_select = 2
m = -1
dist_func = lambda x1, x2 : np.sum(np.abs(x1 - x2)**2, 1)**(1.0/2.0)

# Initialize cross validation strategy.
cv_startegy = StratifiedKFold(n_splits=5)

##########################




### PIPELINES #############

# Initialize pipeline for Relief algorithm
pipeline_relief = Pipeline([('scale', StandardScaler()), 
    ('relief', Relief(n_features_to_select=n_features_to_select, m=m, dist_func=dist_func)), 
    ('classify', RandomForestClassifier(n_estimators=100))])

##########################



### GRID SEARCH ##########

# TODO: include metric learning functions in parameters grid.

# Define parameters grid for grid search.
param_grid = {
    'relief__n_features_to_select': np.arange(1, data.shape[1]+1),
    'relief__dist_func': [manhattan, euclidean],
    'relief__learned_metric_func' : [None, me_dissim_dist_func, covariance_dist_func]
}

# Initialize grid search.
grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=cv_startegy, n_jobs=-1)

# Perform grid search for best hyperparameters.
results = grid_search.fit(data, target)

##########################
