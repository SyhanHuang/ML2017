import os
from keras.models import load_model
import keras.backend as K
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


base_dir = './'
img_dir = os.path.join(base_dir, 'image')
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
cmap_dir = os.path.join(img_dir, 'cmap')
if not os.path.exists(cmap_dir):
    os.makedirs(cmap_dir)
partial_see_dir = os.path.join(img_dir,'partial_see')
if not os.path.exists(partial_see_dir):
    os.makedirs(partial_see_dir)

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

def deprocess_image(x):
	# normalize tensor: center on 0., ensure std is 0.1
	x -= np.mean(x)
	x /= (np.std(x) + 1e-5)
	x /= np.max(x)
	return x

def main():

	store_path = './'
	model_path = os.path.join(store_path,'model.h5')
	emotion_classifier = load_model(model_path)
	(x_train, y_train) = load_data()
	x_train = x_train[25800:28709]
	y_train = y_train[25800:28709]
	private_pixels = x_train
	private_pixels = [ private_pixels[i].reshape((1, 48, 48, 1)) 
		for i in range(len(private_pixels)) ]
	input_img = emotion_classifier.input
	img_ids = [0]#np.arange(21,51)
	
	for idx in img_ids:
		predictions = emotion_classifier.predict_classes(private_pixels[idx])
		val_proba = emotion_classifier.predict(private_pixels[idx])
		pred = val_proba.argmax(axis=-1)
		target = K.mean(emotion_classifier.output[:, pred])
		grads = K.gradients(target, input_img)[0]
		fn = K.function([input_img, K.learning_phase()], [grads])
		
		input_img_data = fn([private_pixels[idx], 0])

		heatmap = deprocess_image(input_img_data).reshape(48, 48)
		
		thres = 0.00000001
		see = private_pixels[idx].reshape(48, 48)
		print(predictions)
		print(y_train[idx])
		print(np.mean(heatmap))
		see[np.where(heatmap <= thres)] = np.mean(see)

		plt.figure()
		plt.imshow(heatmap, cmap=plt.cm.jet)
		plt.colorbar()
		plt.tight_layout()
		fig = plt.gcf()
		plt.draw()
		fig.savefig(os.path.join(cmap_dir, '{}.png'.format(idx)), dpi=100)

		plt.figure()
		plt.imshow(see*255,cmap='gray')
		plt.colorbar()
		plt.tight_layout()
		fig = plt.gcf()
		plt.draw()
		fig.savefig(os.path.join(partial_see_dir, '{}.png'.format(idx)), dpi=100)

if __name__ == "__main__":
    main()