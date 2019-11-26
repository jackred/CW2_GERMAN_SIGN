from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

from helper import pre_processed_data, pre_processed_label
from arg import rForest_args

import numpy as np
import random

def random_forest(data_train, label_train, data_test):
    clf = RandomForestClassifier(
        n_estimators=1000,
        # max_depth=10,
        max_features=10
        # min_samples_leaf=10
    )
    clf.fit(data_train, label_train)
    return clf.predict(data_test)

if __name__ == "__main__":
    print('start random tree classification')
    args = rForest_args()
    rand = np.random.randint(100000)
    data_train, data_test = pre_processed_data(args, rand)
    label_train, label_test = pre_processed_label(args, rand)
    print('data loaded')
    random_forest(data_train, label_train, data_test)
    print('random tree done')
