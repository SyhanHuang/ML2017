import numpy as np
import time
from sklearn.svm import LinearSVR as SVR
from gen import get_eigenvalues
from scipy import misc

# Train a linear SVR
start = time.time()
npzfile = np.load('svr_train.npz')
X = npzfile['X']
y = npzfile['y']

# we already normalize these values in gen.py
# X /= X.max(axis=0, keepdims=True)

svr = SVR(C=1.25)
svr.fit(X, y)

# predict
row, col = [480, 512]
testdata = []
path = './hand/'
for i in range(1, 482):
    testdata.append(misc.imread(path + 'hand.seq' + str(i) + '.png').reshape(row*col))
testdata = np.array(testdata)

mu = testdata.mean(axis=0, keepdims=True)
testdata_ctr = testdata - mu
u, s, v = np.linalg.svd(testdata_ctr)
reduceddata = np.dot(testdata_ctr, v[0:100].T)

test_X = []
data = reduceddata.T
vs = get_eigenvalues(data)
test_X.append(vs)

test_X = np.array(test_X)
pred_y = svr.predict(test_X)
pred_y[np.where(pred_y <= 1)] = 1
pred_y = np.round(pred_y)

with open('ans.csv', 'w') as f:
    print('SetId,LogDim', file=f)
    for i, d in enumerate(pred_y):
        print(f'{i},{np.log(d)}', file=f)

end = time.time()
print('elapsed time = ', end-start)
