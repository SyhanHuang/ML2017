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
            writer.writerow((str(i+1), str(int(predict[i][0]))))
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

def shuffle(X, Y):
    np.random.seed(0)
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])

def addLayer(neuronNum, featureNum):
    np.random.seed(0)
    w = np.random.rand(neuronNum, featureNum)
    w_grad = np.zeros((neuronNum, featureNum))
    m = np.zeros((neuronNum, featureNum))
    v = np.zeros((neuronNum, featureNum))
    b = np.random.rand(1)
    b_grad = np.zeros(1)
    bm = np.zeros(1)
    bv = np.zeros(1)
    return w, w_grad, m, v, b, b_grad, bm, bv

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
#idx = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 16, 20, 22, 23, 24, 25, 26, 27, 29, 30, 31, 33, 35, 38, 40, 41, 42, 43, 44, 45, 47, 49, 50, 51, 53, 54, 56, 57, 58, 61, 63, 102, 105]
[x_train1, x_test1]= [x_train, x_test]
# Parameters
lr = 0.002
l_reg = 0.01
epoch = 25
batch = 50
batchNum = int(np.floor(len(x_train1) / batch))
l_history = []
[w1, w_grad1, m1, v1, b1, b_grad1, bm1, bv1] = addLayer(1, 106)
# Optimizer = Adam
beta1 = 0.9 
beta2 = 0.999
epsilon = 1e-8

for i in range(0, epoch):
    [x_train1, y_train] = shuffle(x_train1, y_train)
    loss = 0.0
    for j in range(1, batchNum+1):
        x_train_batch = x_train1[j*batch:(j+1)*batch]
        y_train_batch = y_train[j*batch:(j+1)*batch]
        # hidden layer1
        z1 = (np.dot(x_train_batch, w1.T) + b1).reshape(len(x_train_batch),)
        y1 = sigmoid(z1)
        loss += (-1.0) * (np.dot(y_train_batch, np.log(y1)) + np.dot((1 - y_train_batch), np.log(1 - y1))).reshape(1,)
        # gradient descent
        w_grad1 = np.dot((y1 - y_train_batch), x_train_batch) + w1 * l_reg
        b_grad1 = np.sum((y1 - y_train_batch))
        # optimizer = adam
        w_lr1 = lr * np.sqrt(1 - beta2 ** j) / (1 - beta1 ** j)
        m1 = beta1 * m1 + (1 - beta1) * w_grad1
        v1 = beta2 * v1 + (1 - beta2) * (w_grad1 ** 2)
        w1 = w1 - w_lr1 * m1 / (np.sqrt(v1) + epsilon)
        bm1 = beta1 * bm1 + (1 - beta1) * b_grad1
        bv1 = beta2 * bv1 + (1 - beta2) * (b_grad1 ** 2)
        b1 = b1 - w_lr1 * b1 / (np.sqrt(bv1) + epsilon)
        
    # store parameters
    l_history.append(loss)

#print("accuracy : ", accuracy(np.around(sigmoid(np.dot(x_train1, w1.T) + b1)), y_train))   

y_test = sigmoid(np.dot(x_test1, w1.T) + b1)
predict = np.around(y_test)
writeFile(predict)

"""
plt.plot(l_history)
plt.title('loss_history')
plt.ylabel('Cross Entropy')
plt.xlabel('epoch #')
plt.legend(['loss'], loc='upper right')
plt.tight_layout()
plt.show()       
"""
