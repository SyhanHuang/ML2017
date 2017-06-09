from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, Dot, dot, Add, Concatenate, Embedding
from keras.callbacks import Callback


class History(Callback):
    def on_train_begin(self,logs={}):
        self.tr_losses=[]
        self.val_losses=[]

    def on_epoch_end(self,epoch,logs={}):
        self.tr_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

def MF(n_users, n_items, latent_dim):

	user_input = Input(shape=[1])
	item_input = Input(shape=[1])
	user_vec = Embedding(n_users, latent_dim, embeddings_initializer= 'random_normal')(user_input)
	user_vec = Flatten()(user_vec)
	item_vec = Embedding(n_items, latent_dim, embeddings_initializer= 'random_normal')(item_input)
	item_vec = Flatten()(item_vec)

	user_bias = Embedding(n_users, 1, embeddings_initializer= 'zeros')(user_input)
	user_bias = Flatten()(user_bias)
	item_bias = Embedding(n_items, 1, embeddings_initializer= 'zeros')(item_input)
	item_bias = Flatten()(item_bias)

	r_hat = Dot(axes= 1)([user_vec, item_vec])
	r_hat = Add()([r_hat, user_bias, item_bias])
	model = Model([user_input, item_input], r_hat)
	model.compile(loss= 'mse', optimizer= 'adamax')
	
	return model

def DNN(n_users, n_items, latent_dim):

	user_input = Input(shape=[1])
	item_input = Input(shape=[1])
	user_vec = Embedding(n_users, latent_dim, embeddings_initializer= 'random_normal')(user_input)
	user_vec = Flatten()(user_vec)
	item_vec = Embedding(n_items, latent_dim, embeddings_initializer= 'random_normal')(item_input)
	item_vec = Flatten()(item_vec)
	Merge_vec = Concatenate()([user_vec, item_vec])
	hidden = Dense(256, activation= 'relu')(Merge_vec)
	hidden = Dropout(0.2)(hidden)
	hidden = Dense(128, activation= 'relu')(hidden)
	hidden = Dropout(0.2)(hidden)
	output = Dense(1)(hidden)
	model = Model([user_input, item_input], output)
	model.compile(loss= 'mse', optimizer= 'adam')
	model.summary()
	
	return model