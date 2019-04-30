import numpy as np
import sys, pdb, random

def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte'
                               % kind)

    with open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

def k1(v1,v2,degree):
	op = np.dotproduct(v1, v2)
	op=np.pow(op,degree)
	return op
	
def mercer_pegasos(kernel,data,labels,T, lamda=1.0):
	alpha = np.zeros((T))
	S=len(data)
	for k in range(1,T):
		it = random.randrange(S)
		sum=0
		for j in range(0,S):
			sm+=alpha[j]*labels[it]*k1(data[it],data[j])
		if labels[it]*(1/(k*lamda))*sm <1:
			alpha[it] = alpha[it]+1
	return alpha


X_train, y_train = load_mnist('../fashionmnist/', kind='train')
X_test, y_test = load_mnist('../fashionmnist/', kind='t10k')

X_train_binary=[]
y_train_binary=[]
X_test_binary =[]
y_test_binary =[]
for k in range(0,y_train.shape[0]):
	if y_train[k]==1:
		X_train_binary.append(X_train[k])
		y_train_binary.append(1)
	elif y_train[k]==2:
		X_train_binary.append(X_train[k])
		y_train_binary.append(-1)
for j in range(0,y_test.shape[0]):
	if y_test[j]==1:
		X_test_binary.append(X_test[j])
		y_test_binary.append(1)
	elif y_test[j]==2:
		X_test_binary.append(X_test[j])
		y_test_binary.append(-1)
print("Number of train set samples for binary case :",len(X_train_binary))
print("Number of test set samples for binary case :",len(X_test_binary))