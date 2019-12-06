from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

from helper import pre_processed_data_all, pre_processed_label_all, print_result, tree_to_png, run_function, extract_measures, plot_experiment_server
from arg import rForest_args

import numpy as np
import random

def random_forest(data_train, label_train, data_test, n_estimator):
    clf = RandomForestClassifier(
        n_estimators=n_estimator,
        max_features=10
    )
    clf.fit(data_train, label_train)
    return clf.predict(data_test)# , clf.predict_proba(data_test)

def server():
	args = rForest_args()
	rand = np.random.randint(10000000)
	data_train, data_test = pre_processed_data_all(args, rand)
	label_train, label_test = pre_processed_label_all(args, rand)
	res = []
	for i in range(30):
		print('===\n=====Epochs: %d=====\n===' % i)
		res.append(run_function(random_forest,
					args.cross_validate,
					data_train, label_train,
					data_test, label_test, n_estimator=i+1))
	print(res)
	res = extract_measures(res)
	print(res)
	plot_experiment_server('random_forest_test_9000', 'n estimator', res)

if __name__ == "__main__":
    server()

    # print('start random tree classification')
    # args = rForest_args()
    # rand = np.random.randint(100000)
    # data_train, data_test = pre_processed_data_all(args, rand)
    # label_train, label_test = pre_processed_label_all(args, rand)
    # print('data loaded')
    # found, confidence = random_forest(data_train, label_train, data_test)
    # print('random tree done')
    # compare_class_true_positive(found, label_test)
    # compare_class(found, label_test)
    # measure(found, label_test, confidence, True)
