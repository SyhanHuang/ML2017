from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten, ZeroPadding2D, advanced_activations, normalization
from keras.optimizers import SGD, Adam

nb_class = 7

class History(Callback):
	def on_train_begin(self,logs={}):
		self.tr_losses=[]
		self.val_losses=[]
		self.tr_accs=[]
		self.val_accs=[]

	def on_epoch_end(self,epoch,logs={}):
		self.tr_losses.append(logs.get('loss'))
		self.val_losses.append(logs.get('val_loss'))
		self.tr_accs.append(logs.get('acc'))
		self.val_accs.append(logs.get('val_acc'))

def vgg():

	model = Sequential()
	model.add(ZeroPadding2D(padding=(1, 1),input_shape=(48, 48, 1)))
	model.add(Convolution2D(64, 3, 3, name='conv1_1'))
	model.add(advanced_activations.PReLU(init='zero', weights=None))
	model.add(normalization.BatchNormalization())
	model.add(ZeroPadding2D(padding=(1, 1)))
	model.add(Convolution2D(64, 3, 3, name='conv1_2'))
	model.add(advanced_activations.PReLU(init='zero', weights=None))
	model.add(normalization.BatchNormalization())
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))
	model.add(Dropout(0.2))

	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(128, 3, 3, name='conv2_1'))
	model.add(advanced_activations.PReLU(init='zero', weights=None))
	model.add(normalization.BatchNormalization())
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(128, 3, 3, name='conv2_2'))
	model.add(advanced_activations.PReLU(init='zero', weights=None))
	model.add(normalization.BatchNormalization())
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))
	model.add(Dropout(0.2))

	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(256, 3, 3, name='conv3_1'))
	model.add(advanced_activations.PReLU(init='zero', weights=None))
	model.add(normalization.BatchNormalization())
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(256, 3, 3, name='conv3_2'))
	model.add(advanced_activations.PReLU(init='zero', weights=None))
	model.add(normalization.BatchNormalization())
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))
	model.add(Dropout(0.2))

	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, 3, 3, name='conv4_1'))
	model.add(advanced_activations.PReLU(init='zero', weights=None))
	model.add(normalization.BatchNormalization())
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(512, 3, 3, name='conv4_2'))
	model.add(advanced_activations.PReLU(init='zero', weights=None))
	model.add(normalization.BatchNormalization())
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))
	model.add(Dropout(0.2))

	model.add(Flatten())
	model.add(Dense(1024, name='fc5'))
	model.add(advanced_activations.PReLU(init='zero', weights=None))
	model.add(normalization.BatchNormalization())
	model.add(Dropout(0.2))
	model.add(Dense(nb_class, activation='softmax', name='fc6'))
	model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])

	return model
	
def dnn():

	model = Sequential()
	model.add(Dense(64, input_shape=(48*48, 1)))
	model.add(Dense(64, name='fc1'))
	model.add(advanced_activations.PReLU(init='zero', weights=None))
	model.add(normalization.BatchNormalization())
	model.add(Dropout(0.2))
	model.add(Dense(64, name='fc2'))
	model.add(advanced_activations.PReLU(init='zero', weights=None))
	model.add(normalization.BatchNormalization())
	model.add(Dropout(0.2))
	model.add(Dense(64, name='fc3'))
	model.add(advanced_activations.PReLU(init='zero', weights=None))
	model.add(normalization.BatchNormalization())
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(64, name='fc4'))
	model.add(advanced_activations.PReLU(init='zero', weights=None))
	model.add(normalization.BatchNormalization())
	model.add(Dropout(0.2))
	model.add(Dense(nb_class, activation='softmax', name='fc5'))
	model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])

	return model