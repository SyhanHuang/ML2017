import numpy as np
import csv
import sys

#start = time.time()

trainFile = sys.argv[1]
testFile = sys.argv[2]
output = sys.argv[3]

# read trainning data
text = open(trainFile, "r", encoding="big5")
df = csv.reader(text, delimiter = ",")

rawData = []
testData = []
x_data = []
y_data = []
testIn = []

for i in range(18):
    rawData.append([])
    testData.append([])

row = 0
for i in df:
    if row != 0:
        for j in range(3, 27):
            if i[j] == "NR":
                rawData[(row-1)%18].append(float(0))
            else:
                rawData[(row-1)%18].append(float(i[j]))
    row += 1
text.close()

for i in range(12):
    for j in range(471):
        x_data.append([1])
        for s in range(9):
            x_data[471*i+j].append(rawData[9][480*i+j+s])
        y_data.append(rawData[9][480*i+j+9])

# read testing data
text = open(testFile, "r")
df = csv.reader(text, delimiter = ",")
row = 0
for i in df:
    for j in range(2, 11):
        if i[j] == "NR":
            testData[(row)%18].append(float(0))
        else:
            testData[(row)%18].append(float(i[j]))
    row += 1
text.close()

for i in range(240):
    testIn.append([1])
    for s in range(9):
        testIn[i].append(testData[9][9*i+s])

# linear regression
# Y = b + W1*X1 + W2*X2 + W3*X3 + ... + W163*X163
x_data = np.asarray(x_data)
y_data = np.asarray(y_data)
testData = np.asarray(testData)
np.random.seed(0)
w = np.random.rand(10)/100
w_lr = np.zeros(10)
w_grad = np.zeros(10)
# for regularization
lamda = 0.001
# for Adam
w_lr1 = 0
m = np.zeros(10)
v = np.zeros(10)
beta1 = 0.9 
beta2 = 0.999
epsilon = 1e-8
# learning rate
lr = 0.0001
iteration = 600
# Store initial values for plotting.
w_history = [w]
l_history = []

# Iterations
for i in range(iteration):
    for j in range(len(x_data)):
        w0 = w
        w0[0] = 0
        loss_temp = y_data[j] - np.dot(x_data[j], w) #+ lamda * np.sum(w**2)
        w_grad = (-2.0) * loss_temp * x_data[j] + (2 * lamda * w0)
        """
        #AdaGrad
        loss_temp = y_data - np.dot(x_data, w) #+ lamda * np.sum(w**2)
        w_grad = -2.0 * np.dot(x_data.T, loss_temp)
        w_lr = w_lr + np.power(w_grad, 2)
        w = w - (lr/np.sqrt(w_lr)) * w_grad  
        """
        #Adam
        if i != 0 :
            w_lr1 = lr * np.sqrt(1 - beta2 ** i) / (1 - beta1 ** i)
            m = beta1 * m + (1 - beta1) * w_grad
            v = beta2 * v + (1 - beta2) * w_grad**2
            w = w - w_lr1 * m / (np.sqrt(v) + epsilon)
    # Store parameters for plotting
    w_history.append(w)
    l_history.append(np.sqrt(np.mean((y_data - np.dot(x_data, w))**2)))

"""
#plot result
plt.plot(l_history)
plt.title('loss_history')
plt.ylabel('RMSE loss')
plt.xlabel('iteration times')
plt.legend(['loss'], loc='upper right')
plt.tight_layout()
plt.show()

loss = np.sqrt(np.mean((y_data - np.dot(x_data, w))**2))
print("loss = ", loss)

end = time.time()
print("run time = ", end-start)
"""
predict = np.dot(testIn, w)
# output csv file
ofile = open(output, "w", newline="")
try:
    writer = csv.writer(ofile)
    writer.writerow(("id", "value"))
    for i in range(240):
        writer.writerow(("id_"+str(i), predict[i]))
finally:
    ofile.close()
