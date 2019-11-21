from sklearn import datasets
from sklearn import metrics
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from src.helper import pre_processed_data, pre_processed_label
from src.arg import parse_args


if __name__ == '__main__':
	args = parse_args("kmeans").parse_args(["-r", "-s", "0.1"])
	rand = np.random.randint(0, 10000000)
	print("Fetching data:")
	data, testdata = pre_processed_data(args, rand)
	print("Done")
	print("Fetching labels:")
	label, testlabel = pre_processed_label(args, rand)

	# load the iris datasets
	dataset = datasets.load_iris()
	print(dataset)
	# fit a CART model to the data
	model = DecisionTreeClassifier()
	model.fit(data, label)
	print(model)
	# make predictions
	expected = testlabel
	predicted = model.predict(testdata)
	# summarize the fit of the model
	print(metrics.classification_report(expected, predicted))
	print(metrics.confusion_matrix(expected, predicted))