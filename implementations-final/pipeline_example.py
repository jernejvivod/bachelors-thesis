import numpy as np
from sklearn.pipeline import Pipeline
from relief import Relief
from relieff import Relieff
from iterative_relief import IterativeRelief
from irelief import IRelief
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

manhattan_w = lambda w, x1, x2 : np.sum(np.abs(w*(x1-x2)), 1)
manhattan_w.kind = "Manhattan (weighted)"

euclidean_w = lambda w, x1, x2: np.sum(np.abs(w*(x1-x2))**2, 1)**(1.0/2.0)
euclidean_w.kind = 'Euclidean (weighted)'

manhattan_w_s = lambda w, x1, x2 : np.sum(np.abs(w*(x1-x2)))
manhattan_w_s.kind = "Manhattan (weighted)"

euclidean_w_s = lambda w, x1, x2: np.sum(np.abs(w*(x1-x2))**2)**(1.0/2.0)
euclidean_w_s.kind = 'Euclidean (weighted)'

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
k = 5
dist_func = lambda x1, x2 : np.sum(np.abs(x1 - x2)**2, 1)**(1.0/2.0)
learned_metric_func = None
max_iter = 100
min_incl = 3
k_width = 3
initial_w_div = 1
conv_condition = 1.0e-9

# Initialize cross validation strategy.
cv_startegy = StratifiedKFold(n_splits=5)

##########################




### PIPELINES #############

# Initialize pipeline for Relief algorithm
pipeline_relief = Pipeline([('scale', StandardScaler()), 
    ('relief', Relief(n_features_to_select=n_features_to_select, m=m, dist_func=dist_func)), 
    ('classify', RandomForestClassifier(n_estimators=100))])

pipeline_relieff = Pipeline([('scale', StandardScaler()), 
    ('relieff', Relieff(n_features_to_select=n_features_to_select, m=m, k=k, dist_func=dist_func)), 
    ('classify', RandomForestClassifier(n_estimators=100))])

pipeline_iterative_relief = Pipeline([('scale', StandardScaler()), 
    ('iterative_relief', IterativeRelief(n_features_to_select=n_features_to_select, m=m, min_incl=min_incl, dist_func=dist_func, max_iter=max_iter)), 
    ('classify', RandomForestClassifier(n_estimators=100))])

pipeline_irelief = Pipeline([('scale', StandardScaler()), 
    ('irelief', IRelief(dist_func=lambda x1, x2 : np.sum(np.abs(x1-x2)), max_iter=max_iter, k_width=k_width, conv_condition=conv_condition, initial_w_div=initial_w_div)), 
    ('classify', RandomForestClassifier(n_estimators=100))])

##########################



### GRID SEARCH ##########

# Define parameters grid for grid search.
param_grid_relief = {
    'relief__n_features_to_select': np.arange(1, data.shape[1]+1),
    'relief__dist_func': [manhattan, euclidean],
    'relief__learned_metric_func' : [None, covariance_dist_func]
}

param_grid_relieff = {
    'relieff__n_features_to_select': np.arange(1, data.shape[1]+1),
    'relieff__dist_func': [manhattan, euclidean],
    'relieff__learned_metric_func' : [None, covariance_dist_func]
}

param_grid_iterative_relief = {
    'iterative_relief__n_features_to_select': np.arange(1, data.shape[1]+1),
    'iterative_relief__dist_func' : [manhattan_w, euclidean_w],
    'iterative_relief__min_incl' : np.arange(1, 3),
    'iterative_relief__learned_metric_func' : [None, covariance_dist_func]
}

param_grid_irelief = {
    'irelief__n_features_to_select': np.arange(1, data.shape[1]+1),
    'irelief__dist_func' : [manhattan_w_s, euclidean_w_s],
    'irelief__learned_metric_func' : [None, covariance_dist_func]
}

# Initialize grid search.
grid_search_relief = GridSearchCV(pipeline_relief, param_grid=param_grid_relief, cv=cv_startegy, verbose=True, n_jobs=-1)
grid_search_relieff = GridSearchCV(pipeline_relieff, param_grid=param_grid_relieff, cv=cv_startegy, verbose=True, n_jobs=-1)
grid_search_iterative_relief = GridSearchCV(pipeline_iterative_relief, param_grid=param_grid_iterative_relief, cv=cv_startegy, verbose=True)
grid_search_irelief = GridSearchCV(pipeline_irelief, param_grid=param_grid_irelief, cv=cv_startegy, verbose=True, n_jobs=-1)

# Perform grid search for best hyperparameters.
res_relief = grid_search_relief.fit(data, target)
res_relieff = grid_search_relieff.fit(data, target)
res_iterative_relief = grid_search_iterative_relief.fit(data, target)
res_irelief = grid_search_irelief.fit(data, target)

##########################

