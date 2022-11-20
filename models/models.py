from sklearn import datasets, svm, metrics
import numpy as np
import pandas as pd
import pdb

from utils import (
    preprocess_digits,
    train_dev_test_split,
    get_all_h_param_comb,
    tune_and_save,
    macro_f1
)

def models_clf():
    n_cv = 1
    results = {}
    for n in range(n_cv):
        x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(
            data, label, train_frac, dev_frac
        )
        # PART: Define the model
        # Create a classifier: a support vector classifier
        models_of_choice = {
            "svm": svm.SVC(),
            "decision_tree": tree.DecisionTreeClassifier(),
        }
        for clf_name in models_of_choice:
            clf = models_of_choice[clf_name]
            print("[{}] Running hyper param tuning for {}".format(n,clf_name))
            actual_model_path = tune_and_save(
                clf, x_train, y_train, x_dev, y_dev, h_metric, h_param_comb[clf_name], model_path=None
            )

            # 2. load the best_model
            best_model = load(actual_model_path)

            # PART: Get test set predictions
            # Predict the value of the digit on the test subset
            predicted = best_model.predict(x_test)
            if not clf_name in results:
                results[clf_name]=[]    

            results[clf_name].append({m.__name__:m(y_pred=predicted, y_true=y_test) for m in metric_list})
            # 4. report the test set accurancy with that best model.
            # PART: Compute evaluation metrics
            print(
                f"Classification report for classifier {clf}:\n"
                f"{metrics.classification_report(y_test, predicted)}\n"
            )

    print(results)