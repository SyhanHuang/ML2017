import csv
import sys
import numpy as np

def readData():
    X_train = np.delete(np.genfromtxt(sys.argv[3], delimiter=','), 0, 0)
    Y_train = np.genfromtxt(sys.argv[4], delimiter=',')
    X_test = np.delete(np.genfromtxt(sys.argv[5], delimiter=','), 0, 0)
    return X_train, Y_train, X_test

def writeFile(predict):
    ofile = open(sys.argv[6], "w", newline="")
    try:
        writer = csv.writer(ofile)
        writer.writerow(("id", "label"))
        for i in range(len(predict)):
            writer.writerow((str(i+1), str(int(predict[i]))))
    finally:
        ofile.close()

def feature_scaling(X_train, X_test):
	# feature scaling with all X
	X_all = np.concatenate((X_train, X_test))
	mu = np.mean(X_all, axis=0)
	sigma = np.std(X_all, axis=0)
	
	# only apply on continuos attribute
	index = [0, 1, 3, 4, 5]
	mean_vec = np.zeros(X_all.shape[1])
	std_vec = np.ones(X_all.shape[1])
	mean_vec[index] = mu[index]
	std_vec[index] = sigma[index]

	X_all_normed = (X_all - mean_vec) / std_vec

	# split train, test again
	X_train_normed = X_all_normed[0:X_train.shape[0]]
	X_test_normed = X_all_normed[X_train.shape[0]:]

	return X_train_normed, X_test_normed

def sigmoid(z):
    output = 1 / (1 + np.exp(-z))
    return np.clip(output, 0.00000000000001, 0.99999999999999)

def accuracy(y_train, y_hat):
    output = 0
    for i in range(len(y_hat)):
        if y_train[i] == y_hat[i]:
            output += 1/len(y_hat)
    return output

trainData = sys.argv[1]
testData = sys.argv[2]

[x_train, y_train ,x_test] = readData()
[x_train, x_test] = feature_scaling(x_train, x_test)
# Parameters
lr = 0.001
epoch = 100
batch = 50
batchNum = int(np.floor(len(x_train) / batch))
l_history = []
# Gaussian distribution
size1 = 0
size2 = 0
mean1 = np.zeros(x_train.shape[1])
mean2 = np.zeros(x_train.shape[1])
sigma1 = np.zeros((x_train.shape[1], x_train.shape[1]))
sigma2 = np.zeros((x_train.shape[1], x_train.shape[1]))

# Calculate mean
for i in range(x_train.shape[0]):
    if y_train[i] == 1:
        mean1 += x_train[i]
        size1 += 1
    else:
        mean2 += x_train[i]
        size2 += 1
mean1 = mean1 / size1
mean2 = mean2 / size2

# Calculate covariance matrix
for i in range(x_train.shape[0]):
    if y_train[i] == 1:
        sigma1 += np.dot(np.transpose([x_train[i] - mean1]), [(x_train[i] - mean1)])
        size1 += 1
    else:
        sigma2 += np.dot(np.transpose([x_train[i] - mean2]), [(x_train[i] - mean2)])
        size2 += 1
sigma1 = sigma1 / size1
sigma2 = sigma2 / size2
sigma = (size1 / x_train.shape[0]) * sigma1 + (size2 / x_train.shape[0]) * sigma2

# Predict
sigmaInv = np.linalg.inv(sigma)
w = np.dot((mean1 - mean2), sigmaInv)
b = (-0.5) * np.dot(np.dot([mean1], sigmaInv), mean1) + (0.5) * np.dot(np.dot([mean2], sigmaInv), mean2) + np.log(size1 / size2)
z = np.dot(x_test, w) + b

y_test = sigmoid(z)
predict = np.around(y_test)
writeFile(predict)
