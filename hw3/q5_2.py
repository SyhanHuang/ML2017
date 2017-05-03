import os
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
import numpy as np
import pandas as pd

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

def main():
	base_path = './'
	store_path = './'
	model_path = os.path.join(base_path,'model.h5')
	emotion_classifier = load_model(model_path)
	layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[1:])

	input_img = emotion_classifier.input
	name_ls = ['conv1_1']
	collect_layers = [ K.function([input_img, K.learning_phase()], [layer_dict[name].output]) for name in name_ls ]
	
	(x_train, y_train) = load_data()
	x_train = x_train[25800:28709]
	y_train = y_train[25800:28709]
	private_pixels = x_train
	private_pixels = [ private_pixels[i].reshape((1, 48, 48, 1)) 
					   for i in range(len(private_pixels)) ]
	choose_id = 0
	photo = private_pixels[choose_id]
	for cnt, fn in enumerate(collect_layers):
		im = fn([photo, 0]) #get the output of that layer
		fig = plt.figure(figsize=(14, 8))
		nb_filter = im[0].shape[3]
		for i in range(nb_filter):
			ax = fig.add_subplot(nb_filter/16, 16, i+1)
			ax.imshow(im[0][0, :, :, i], cmap='BuGn')
			plt.xticks(np.array([]))
			plt.yticks(np.array([]))
			plt.tight_layout()
		fig.suptitle('Output of layer{} (Given image{})'.format(cnt, choose_id))
		img_path = os.path.join(store_path, 'vis')
		if not os.path.isdir(img_path):
			os.mkdir(img_path)
		fig.savefig(os.path.join(img_path,'layer{}'.format(cnt)))

if __name__ == "__main__":
    main()