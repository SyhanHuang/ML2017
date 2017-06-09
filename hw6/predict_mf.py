import pandas as pd
import numpy as np
import sys
import os
import model

data_path = sys.argv[1]
user_path = os.path.join(data_path, 'users.csv')
movie_path = os.path.join(data_path, 'movies.csv')
test_path = os.path.join(data_path, 'test.csv')
output_path = sys.argv[2]


users = pd.read_csv(user_path, sep='::', 
						engine='python', 
						names=['userid', 'gender', 'age', 'occupation', 'zip'])
test = pd.read_csv(test_path, engine='python', sep=',',
						names=['id', 'userid', 'movieid']).set_index('id')
movies = pd.read_csv(movie_path, engine='python', sep='::',
						names=['movieid', 'title', 'genre'])
movies['genre'] = movies.genre.str.split('|')

users_mat = users.as_matrix()
users_mat = np.delete(users_mat, 0, 0)
movies_mat = movies.as_matrix()
movies_mat = np.delete(movies_mat, 0, 0)

test_mat = test.as_matrix()
test_mat = np.delete(test_mat, 0, 0).astype('float')

num_user = users_mat.shape[0]
num_movie = movies_mat.shape[0]


model = model.MF(num_user, num_movie, 50)
model.load_weights('./best.hdf5')
predict = model.predict([test_mat[:,0], test_mat[:,1]])

with open(output_path,'w') as output:
		print ('\"TestDataID\",\"Rating\"',file=output)
		for index,values in enumerate(predict):
			print ('%d,%f'%(index+1,values),file=output)
