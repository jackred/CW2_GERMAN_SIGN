#!/usr/bin/python3
from sklearn import metrics
import numpy as np
from src.helper import pre_processed_data_all, pre_processed_label_all
from src.arg import j48_args
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import pydotplus


def j48(data_train, label_train, data_predict, depth):
	modul = DecisionTreeRegressor(max_depth=depth)
	modul.fit(data_train, label_train)
	predicted = modul.predict(data_predict)
	predicted = np.array([int(i) for i in predicted])
	return predicted, modul


if __name__ == '__main__':
	className = ["speed limit 60", "speed limit 80", "speed limit 80 lifted", "right of way at crossing", "right of way in general", "give way", "stop", "no speed limit general", "turn right down", "turn left down"]
	featureName = []
	for i in range(0, 2304):
		featureName.append(i)

	args = j48_args()
	rand = np.random.randint(0, 10000000)
	print("Fetching data:")
	data, testdata = pre_processed_data_all(args, rand)
	print("Done")
	print("Fetching labels:")
	label, testlabel = pre_processed_label_all(args, rand)
	print("Done")

	predicted, model = j48(data, label, testdata, args.depth)
	print(predicted)
	print(np.unique(predicted))
	print(metrics.classification_report(testlabel, predicted))
	print(metrics.confusion_matrix(testlabel, predicted))

	print("Saving tree...")
	dot_data = tree.export_graphviz(
				model,
				out_file=None, 
				feature_names=featureName, 
				class_names=className
	)

	graph = pydotplus.graph_from_dot_data(dot_data)
	graph.write_png("tree.png")
	print("DONE")

