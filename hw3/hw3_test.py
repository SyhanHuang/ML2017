import numpy as np
import pandas as pd
import sys
import csv
from keras.models import load_model


def load_data(path):

	x_test = []
	temp = pd.read_csv(path, skiprows = 0)
	for i in range(temp.shape[0]):
		x_test.append(temp.ix[i, 1].split(' '))
	x_test = np.asarray(x_test)
	x_test = x_test.astype('float32')
	x_test = x_test / 255

	return x_test

def writeFile(predict):
	ofile = open(sys.argv[2], "w", newline="")
	try:
		writer = csv.writer(ofile)
		writer.writerow(("id", "label"))
		for i in range(len(predict)):
			writer.writerow((str(i), str(predict[i])))
	finally:
		ofile.close()


x_test = load_data(sys.argv[1])
x_test = x_test.reshape(x_test.shape[0],48,48,1)
model_path = 'model.h5'
emotion_classifier = load_model(model_path)
predict = emotion_classifier.predict_classes(x_test, batch_size = 100)
writeFile(predict)
