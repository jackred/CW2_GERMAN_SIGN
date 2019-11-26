from sklearn import metrics
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from src.helper import pre_processed_data, pre_processed_label
from src.arg import parse_args


if __name__ == '__main__':
	args = parse_args("kmeans").parse_args(["-r", "-s", "0.9"])
	rand = np.random.randint(0, 10000000)
	print("Fetching data:")
	data, testdata = pre_processed_data(args, rand)
	print("Done")
	print("Fetching labels:")
	label, testlabel = pre_processed_label(args, rand)

	model = DecisionTreeClassifier()
	model.fit(data, label)
	expected = testlabel
	predicted = model.predict(testdata)
	print(metrics.classification_report(expected, predicted))
	print(metrics.confusion_matrix(expected, predicted))