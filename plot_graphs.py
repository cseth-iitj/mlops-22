# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause


# PART: library dependencies -- sklear, torch, tensorflow, numpy, transformers

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics, tree
import pdb
from joblib import dump, load
import argparse
import numpy as np
import pandas as pd

from utils import (
    preprocess_digits,
    train_dev_test_split,
    h_param_tuning,
    data_viz,
    pred_image_viz,
    get_all_h_param_comb,
    tune_and_save,
    macro_f1
)

train_frac, dev_frac, test_frac = 0.8, 0.1, 0.1
assert train_frac + dev_frac + test_frac == 1.0

# 1. set the ranges of hyper parameters
gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]

max_depth_list = [2, 10, 20, 50, 100]
min_samples_leaf = [3, 9, 15, 30]
criterion =  ["gini", "entropy"]

params = {}
params["gamma"] = gamma_list
params["C"] = c_list

svm_params = {}
svm_params["gamma"] = gamma_list
svm_params["C"] = c_list
svm_h_param_comb = get_all_h_param_comb(svm_params)

dec_params = {}
dec_params["max_depth"] = max_depth_list
dec_h_param_comb = get_all_h_param_comb(dec_params)

#h_param_comb = get_all_h_param_comb(params)
h_param_comb = {"svm": svm_h_param_comb, "decision_tree": dec_h_param_comb}


# PART: load dataset -- data from csv, tsv, jsonl, pickle
digits = datasets.load_digits()
data_viz(digits)
data, label = preprocess_digits(digits)
# housekeeping
del digits

parser = argparse.ArgumentParser(description='Process argparse...')
#parser.add_argument('--clf_name', dest='clf', action='store_const', const=svm)
#parser.add_argument('--random_state', dest='random_state', action='store_const', const=svm)
parser.add_argument('--clf_name', dest='clf')
parser.add_argument('--random_state', dest='random_state')
args = parser.parse_args()

x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(
    data, label, train_frac, dev_frac
)

# PART: Define the model
# Create a classifier: a support vector classifier
clf = svm.SVC()
# define the evaluation metric
metric_list = metrics.accuracy_score, macro_f1
h_metric = metrics.accuracy_score


actual_model_path = tune_and_save(
    clf, x_train, y_train, x_dev, y_dev, metric, h_param_comb, model_path=None
)


# 2. load the best_model
best_model = load(actual_model_path)

# PART: Get test set predictions
# Predict the value of the digit on the test subset
predicted = best_model.predict(x_test)

pred_image_viz(x_test, predicted)

# 4. report the test set accurancy with that best model.
# PART: Compute evaluation metrics
print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)

