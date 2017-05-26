import numpy as np
import string
import sys
import pickle
import keras.backend as K 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Convolution1D, MaxPooling1D, ZeroPadding1D, Flatten, GRU, normalization
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import initializers
from keras import regularizers

test_path = sys.argv[1]
output_path = sys.argv[2]

#####################
###   parameter   ###
#####################

embedding_dim = 100

################
###   Util   ###
################
def read_data(path,training):
    print ('Reading data from ',path)
    with open(path,'r',encoding = 'utf-8') as f:
    
        tags = []
        articles = []
        tags_list = []
        
        f.readline()
        for line in f:
            if training :
                start = line.find('\"')
                end = line.find('\"',start+1)
                tag = line[start+1:end].split(' ')
                article = line[end+2:]
                
                for t in tag :
                    if t not in tags_list:
                        tags_list.append(t)
               
                tags.append(tag)
            else:
                start = line.find(',')
                article = line[start+1:]
            
            articles.append(article)
            
        if training :
            assert len(tags_list) == 38,(len(tags_list))
            assert len(tags) == len(articles)
    return (tags,articles,tags_list)


###########################
###   custom metrices   ###
###########################
def f1_score(y_true,y_pred):
    thresh = 0.5
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true * y_pred,axis=-1)
    
    precision=tp/(K.sum(y_pred,axis=-1)+K.epsilon())
    recall=tp/(K.sum(y_true,axis=-1)+K.epsilon())
    return K.mean(2*((precision*recall)/(precision+recall+K.epsilon())))

#########################
###   Main function   ###
#########################
def main():

	(_, X_test,_) = read_data(test_path,False)
	with open ('label_mapping.pickle', 'rb') as f:
		tag_list = pickle.load(f)	
	### tokenizer for all data
	with open ('tokenizer.pickle', 'rb') as f:
		tokenizer = pickle.load(f)
	word_index = tokenizer.word_index
	### convert word sequences to index sequence
	test_sequences = tokenizer.texts_to_sequences(X_test)
	### padding to equal length
	max_article_length = 306
	test_sequences = pad_sequences(test_sequences,maxlen=max_article_length)
	num_words = len(word_index) + 1
	### build model
	model = Sequential()
	model.add(Embedding(num_words,
						embedding_dim,
						weights=None,
						input_length=max_article_length,
						trainable=False))
	model.add(GRU(180,kernel_initializer = initializers.glorot_uniform(seed=1),recurrent_initializer = initializers.RandomNormal(mean=0.0, stddev=0.01, seed=1),kernel_regularizer = regularizers.l2(0.001),activation='tanh',dropout=0.35))
	model.add(Dense(256,activation='relu'))
	model.add(Dropout(0.35))
	model.add(Dense(128,activation='relu'))
	model.add(Dropout(0.35))
	model.add(Dense(64,activation='relu'))
	model.add(Dropout(0.35))
	model.add(Dense(38,activation='sigmoid'))
	adam = Adam(lr=0.00125,decay=1e-6,clipvalue=0.2)
	model.compile(loss='categorical_crossentropy',
				  optimizer=adam,
				  metrics=[f1_score])

	model.load_weights('./best.hdf5')
	Y_pred = model.predict(test_sequences)
	thresh = 0.5
	with open(output_path,'w') as output:
		print ('\"id\",\"tags\"',file=output)
		Y_pred_thresh = (Y_pred > thresh).astype('int')
		for index,labels in enumerate(Y_pred_thresh):
			labels = [tag_list[i] for i,value in enumerate(labels) if value==1 ]
			labels_original = ' '.join(labels)
			print ('\"%d\",\"%s\"'%(index,labels_original),file=output)

if __name__=='__main__':
    main()