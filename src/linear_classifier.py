from sklearn.linear_model import SGDClassifier

from helper import pre_processed_data_all, pre_processed_label_all, print_result, tree_to_png, run_function, extract_measures, plot_experiment_server
from arg import rForest_args

import numpy as np
import random

def linear_classifier(data_train, label_train, data_test, max_iter):
    clf = SGDClassifier(loss='modified_huber', n_jobs=6, n_iter_no_change=max_iter)
    clf.fit(data_train, label_train)
    return clf.predict(data_test)#, clf.predict_proba(data_test)


def server():
	args = rForest_args()
	rand = np.random.randint(10000000)
	data_train, data_test = pre_processed_data_all(args, rand)
	label_train, label_test = pre_processed_label_all(args, rand)
	res = []
	for i in range(5):
		print('===\n=====Epochs: %d=====\n===' % i)
		res.append(run_function(linear_classifier,
					args.cross_validate,
					data_train, label_train,
					data_test, label_test, max_iter=(i + 1)))
	print(res)
	res = extract_measures(res)
	print(res)
	plot_experiment_server('linear_classifier_test', 'max iteration ( x 100)', res)

if __name__ == "__main__":
    server()

    # print('start linear classifier')
    # args = rForest_args()
    # rand = np.random.randint(100000)
    # data_train, data_test = pre_processed_data_all(args, rand)
    # label_train, label_test = pre_processed_label_all(args, rand)
    # print('data loaded')
    # found, confidence = linear_classifier(data_train, label_train, data_test)
    # print('linear classifier done')
    # compare_class_true_positive(found, label_test)
    # compare_class(found, label_test)
    # measure(found, label_test, confidence, True)
    # print(true_positive(found, label_test), true_positive(found, label_test, True))
    # print(false_positive(found, label_test), false_positive(found, label_test, True))
