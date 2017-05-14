import numpy as np
import csv
import sys
from sklearn.svm import LinearSVR as SVR
from gen import get_eigenvalues

def writeFile(predict):
    ofile = open(sys.argv[2], "w", newline="")
    try:
        writer = csv.writer(ofile)
        writer.writerow(("SetId", "LogDim"))
        for i in range(len(predict)):
            writer.writerow((str(i), str(predict[i])))
    finally:
        ofile.close()

# Train a linear SVR
npzfile = np.load('svr_train.npz')
X = npzfile['X']
y = npzfile['y']

# we already normalize these values in gen.py
# X /= X.max(axis=0, keepdims=True)

svr = SVR(C=1.25)
svr.fit(X, y)

# predict
testdata = np.load(sys.argv[1])
test_X = []
for i in range(200):
    data = testdata[str(i)]
    vs = get_eigenvalues(data)
    test_X.append(vs)

test_X = np.array(test_X)
pred_y = svr.predict(test_X)
pred_y[np.where(pred_y <= 1)] = 1
pred_y = np.round(pred_y)
writeFile(np.log(pred_y))
