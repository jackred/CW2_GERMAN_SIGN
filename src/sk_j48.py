#!/usr/bin/python3
import numpy as np
from helper import pre_processed_data_all, pre_processed_label_all, print_result, tree_to_png, run_function, extract_measures, plot_experiment_server
from arg import j48_args
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def j48(data_train, label_train, data_predict, depth, min_split, min_leaf, min_weight):
	#modul = DecisionTreeRegressor(max_depth=depth)
	modul = DecisionTreeClassifier(random_state=0, max_depth=depth, min_samples_split=min_split, min_samples_leaf=min_leaf, min_weight_fraction_leaf=min_weight)
	modul.fit(data_train, label_train)
	predicted = modul.predict(data_predict)
	predicted = np.array([int(i) for i in predicted])
	return predicted


def j48_depth(options):
	args = j48_args(options).parse_args(options[1:])
	rand = np.random.randint(10000000)
	data_train, data_test = pre_processed_data_all(args, rand)
	label_train, label_test = pre_processed_label_all(args, rand)
	res = []
	for i in range(24):
		print('===\n=====Epochs: %d=====\n===' % i)
		res.append(run_function(j48,
								args.cross_validate,
								data_train, label_train,
								data_test, label_test,
								depth=i, min_split=2, min_leaf=1, min_weight=0))
	print(res)
	res = extract_measures(res)
	print(res)
	plot_experiment_server("j48_" + options[0] + "max_depth", 'max depth', res)


def j48_min_split(options):
	args = j48_args(options).parse_args(options[1:])
	rand = np.random.randint(10000000)
	data_train, data_test = pre_processed_data_all(args, rand)
	label_train, label_test = pre_processed_label_all(args, rand)
	res = []
	for i in range(9):
		print('===\n=====Epochs: %d=====\n===' % i)
		res.append(run_function(j48,
								args.cross_validate,
								data_train, label_train,
								data_test, label_test,
								depth=None, min_split=i/10 + 0.1, min_leaf=1, min_weight=0))
	print(res)
	res = extract_measures(res)
	print(res)
	plot_experiment_server("j48_" + options[0] + "min_split", 'min samples split', res)


def j48_min_leaf(options):
	args = j48_args(options).parse_args(options[1:])
	rand = np.random.randint(10000000)
	data_train, data_test = pre_processed_data_all(args, rand)
	label_train, label_test = pre_processed_label_all(args, rand)
	res = []
	for i in range(9):
		print('===\n=====Epochs: %d=====\n===' % i)
		res.append(run_function(j48,
								args.cross_validate,
								data_train, label_train,
								data_test, label_test,
								depth=None, min_split=2, min_leaf=i/10 + 0.1, min_weight=0))
	print(res)
	res = extract_measures(res)
	print(res)
	plot_experiment_server("j48_" + options[0] + "min_leaf", 'min samples leaf', res)


def j48_min_weight(options):
	args = j48_args(options).parse_args(options[1:])
	rand = np.random.randint(10000000)
	data_train, data_test = pre_processed_data_all(args, rand)
	label_train, label_test = pre_processed_label_all(args, rand)
	res = []
	for i in range(4):
		print('===\n=====Epochs: %d=====\n===' % i)
		res.append(run_function(j48,
								args.cross_validate,
								data_train, label_train,
								data_test, label_test,
								depth=None, min_split=2, min_leaf=1, min_weight=i /10 + 0.1))
	print(res)
	res = extract_measures(res)
	print(res)
	plot_experiment_server("j48_" + options[0] + "min_weight", 'min samples leaf', res)


def server():
	options = [
		["cross", "-cross", "10"],
		["testing", "-t"],
		["testing4000", "-t", "-s", "8660"],
		["testing9000", "-t", "-s", "3660"]
	]

	for option in options:
		j48_depth(option)
		j48_min_split(option)
		j48_min_leaf(option)
		j48_min_weight(option)


if __name__ == '__main__':
	server()
	"""args = j48_args()
	rand = np.random.randint(0, 10000000)
	data, testdata = pre_processed_data_all(args, rand)
	label, testlabel = pre_processed_label_all(args, rand)

	predicted, model = j48(data, label, testdata, args.depth)

	print_result(testlabel, predicted)
	tree_to_png(model)"""

