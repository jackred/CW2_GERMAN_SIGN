from sklearn.linear_model import SGDClassifier

from helper import pre_processed_data_all, pre_processed_label_all, \
    compare_class_true_positive, compare_class, measure, true_positive, false_positive
from arg import rForest_args

import numpy as np
import random

def linear_classifier(data_train, label_train, data_test):
    clf = SGDClassifier(loss='modified_huber')
    clf.fit(data_train, label_train)
    return clf.predict(data_test), clf.predict_proba(data_test)

if __name__ == "__main__":
    print('start linear classifier')
    args = rForest_args()
    rand = np.random.randint(100000)
    data_train, data_test = pre_processed_data_all(args, rand)
    label_train, label_test = pre_processed_label_all(args, rand)
    print('data loaded')
    found, confidence = linear_classifier(data_train, label_train, data_test)
    print('linear classifier done')
    compare_class_true_positive(found, label_test)
    compare_class(found, label_test)
    measure(found, label_test, confidence, True)
    print(true_positive(found, label_test), true_positive(found, label_test, True))
    print(false_positive(found, label_test), false_positive(found, label_test, True))
