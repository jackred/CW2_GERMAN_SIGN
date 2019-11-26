from sklearn.linear_model import SGDClassifier

from helper import pre_processed_data, pre_processed_label, compare_class_true_positive, compare_class
from arg import rForest_args

import numpy as np
import random

def linear_classifier(data_train, label_train, data_test):
    clf = SGDClassifier()
    clf.fit(data_train, label_train)
    return clf.predict(data_test)

if __name__ == "__main__":
    print('start random tree classification')
    args = rForest_args()
    rand = np.random.randint(100000)
    data_train, data_test = pre_processed_data(args, rand)
    label_train, label_test = pre_processed_label(args, rand)
    print('data loaded')
    found = linear_classifier(data_train, label_train, data_test)
    print('linear classifier done')
    compare_class_true_positive(found, label_test)
    compare_class(found, label_test)
