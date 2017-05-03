import numpy as np
import pandas as pd
import csv
import os
import sys
import model

from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard

def load_data():
	x_train = []
	temp = pd.read_csv(sys.argv[1], skiprows = 0)
	y_train = np.array(temp.ix[:, 0])
	for i in range(temp.shape[0]):
		x_train.append(temp.ix[i, 1].split(' '))
	x_train = np.asarray(x_train)
	x_train = x_train.astype('float32')
	x_train = x_train / 255
	y_train = y_train.astype('int')
	y_train = np_utils.to_categorical(y_train, 7)

	return (x_train, y_train)

def dump_history(store_path,logs):
    with open(os.path.join(store_path,'train_loss'),'a') as f:
        for loss in logs.tr_losses:
            f.write('{}\n'.format(loss))
    with open(os.path.join(store_path,'train_accuracy'),'a') as f:
        for acc in logs.tr_accs:
            f.write('{}\n'.format(acc))
    with open(os.path.join(store_path,'valid_loss'),'a') as f:
        for loss in logs.val_losses:
            f.write('{}\n'.format(loss))
    with open(os.path.join(store_path,'valid_accuracy'),'a') as f:
        for acc in logs.val_accs:
            f.write('{}\n'.format(acc))

(x_train, y_train) = load_data()

x_val = x_train[25800:28709]
y_val = y_train[25800:28709]
x_train = x_train[0:25800]
y_train = y_train[0:25800]

x_train = x_train.reshape(x_train.shape[0],48,48,1)
x_val = x_val.reshape(x_val.shape[0],48,48,1)

datagen = ImageDataGenerator(
	featurewise_center=False,
	samplewise_center=False,
    featurewise_std_normalization=False,
	samplewise_std_normalization=False,
    rotation_range=0,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
	vertical_flip=True)

datagen.fit(x_train)

emotion_classifier = model.dnn()
emotion_classifier.summary()

history = model.History()
tbCallBack = TensorBoard(log_dir=os.path.join('./','logs'), write_graph=True, write_images=False)
emotion_classifier.fit_generator(datagen.flow(x_train, y_train, batch_size=100),
	steps_per_epoch=len(x_train) / 100, epochs=80, validation_data=(x_val,y_val), callbacks=[history, tbCallBack])
dump_history('./',history)
emotion_classifier.save('model.h5')

score = emotion_classifier.evaluate(x_train,y_train)
print ('Train Acc:', score[1])

