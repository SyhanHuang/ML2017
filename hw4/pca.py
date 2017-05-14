import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

def plot_figure(arr, margin, n, row, col):
    width = n * col + (n - 1) * margin
    height = n * row + (n - 1) * margin
    fig = np.zeros((width, height))
    for i in range(n):
        for j in range(n):
            fig[(col + margin) * i: (col + margin) * i + col,
                (row + margin) * j: (row + margin) * j + row] = arr[i * n + j].reshape(row, col)
    return fig
    
# read file
path = './faceExpressionDatabase/'
name = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
row, col = [64, 64]
img = []
for i in name:
    for j in range(10):
        img.append(misc.imread(path + i + '0' + str(j) + '.bmp').reshape(row*col))
img = np.array(img)
       
mu = img.mean(axis=0, keepdims=True)
img_ctr = img - mu
u, s, v = np.linalg.svd(img_ctr)

# plot top 9 eigenFace
eigenFace = plot_figure(v, 0, 3, row, col)
plt.figure()
plt.imshow(mu.reshape(row, col), cmap=plt.get_cmap('gray'))
plt.colorbar()
plt.title('Average face')
plt.figure()
plt.imshow(eigenFace, cmap=plt.get_cmap('gray'))
plt.colorbar()
plt.title('Top 9 eigenface')

# project 100 faces onto top 5 eigenfaces
originalFace = img[0:100]
reduceFace = np.dot(img_ctr[0:100], v[0:5].T)
reconverFace = np.dot(reduceFace, v[0:5]) + mu

originalFace_img = plot_figure(originalFace, 0, 10, row, col)
recoverFace_img = plot_figure(reconverFace, 0, 10, row, col)

plt.figure()
plt.imshow(originalFace_img, cmap=plt.get_cmap('gray'))
plt.colorbar()
plt.title('Original face')
plt.figure()
plt.imshow(recoverFace_img, cmap=plt.get_cmap('gray'))
plt.colorbar()
plt.title('Project onto top 5 eigenface')

# reconstruction error test
for i in range(0,100):
    reduceFace = np.dot(img_ctr[0:100], v[0:i].T)
    recoverFace = np.dot(reduceFace, v[0:i]) + mu
    rmse = np.sqrt(np.mean(np.mean((originalFace - recoverFace) ** 2, axis = 0), axis = 0)) / 255
    print('%dth rmse = %f' % (i, rmse))
    if rmse < 0.01:
        break

plt.show()
