import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_iris
from preprocessing import dataset_preprocessing
from function.relief import relief
import pdb

# Load data
mpg = dataset_preprocessing.get_tidy_data('./preprocessing/data/auto-mpg.gz')
pdb.set_trace()
iris = load_iris()

# Perform feature selection using relief.
sel_features1 = relief(mpg.data, mpg.target)
sel_features2 = relief(iris.data, iris.target)

# Define classifier.
clf = AdaBoostClassifier(n_estimators=100)

# Perform cross-validation.
scores1 = cross_val_score(clf, sel_features1, mpg.target, cv=10)
mean1 = scores1.mean()
scores2 = cross_val_score(clf, sel_features2, iris.target, cv=5)
mean2 = scores2.mean()