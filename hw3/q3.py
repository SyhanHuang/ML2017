from keras.models import load_model
from sklearn.metrics import confusion_matrix
import os
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt

def load_data():
	x_train = []
	temp = pd.read_csv("train.csv", skiprows = 0)
	y_train = np.array(temp.ix[:, 0])
	for i in range(temp.shape[0]):
		x_train.append(temp.ix[i, 1].split(' '))
	x_train = np.asarray(x_train)
	x_train = x_train.astype('float32')
	x_train = x_train / 255
	y_train = y_train.astype('int')

	
	return (x_train, y_train)

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def main():
	model_path = os.path.join('./','model.h5')
	emotion_classifier = load_model(model_path)
	np.set_printoptions(precision=2)
	(x_train, y_train) = load_data()
	x_val = x_train[25800:28709]
	x_val = x_val.reshape(x_val.shape[0],48,48,1)
	y_val = y_train[25800:28709]
	predictions = emotion_classifier.predict_classes(x_val)
	conf_mat = confusion_matrix(y_val,predictions)
	print(conf_mat)
	plt.figure()
	plot_confusion_matrix(conf_mat, classes=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"])
	plt.show()

if __name__ == "__main__":
    main()