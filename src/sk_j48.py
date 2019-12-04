#!/usr/bin/python3
import numpy as np
from src.helper import pre_processed_data_all, pre_processed_label_all, print_result, tree_to_png, run_function, extract_measures, plot_experiment_server
from src.arg import j48_args
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def j48(data_train, label_train, data_predict, depth):
	#modul = DecisionTreeRegressor(max_depth=depth)
	modul = DecisionTreeClassifier(random_state=0, max_depth=depth)
	modul.fit(data_train, label_train)
	predicted = modul.predict(data_predict)
	predicted = np.array([int(i) for i in predicted])
	return predicted


def server():
	args = j48_args()
	rand = np.random.randint(10000000)
	data_train, data_test = pre_processed_data_all(args, rand)
	label_train, label_test = pre_processed_label_all(args, rand)
	res = []
	for i in range(30):
		print('===\n=====Epochs: %d=====\n===' % i)
		res.append(run_function(j48,
								args.cross_validate,
								data_train, label_train,
								data_test, label_test,
								depth=i+1))
	print(res)
	res = extract_measures(res)
	print(res)
	plot_experiment_server('j48', 'max depth', res)


if __name__ == '__main__':
	server()
	"""args = j48_args()
	rand = np.random.randint(0, 10000000)
	data, testdata = pre_processed_data_all(args, rand)
	label, testlabel = pre_processed_label_all(args, rand)

	predicted, model = j48(data, label, testdata, args.depth)

	print_result(testlabel, predicted)
	tree_to_png(model)"""

