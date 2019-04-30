import numpy as np
import sys, random, pdb

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

def loss(w,x,y):
	op = y*np.dot(w,x)
	return op

def pegasos(data,labels,T,lamda=1.0):
	S = len(data)
	w = np.zeros((data[0].shape[0]))
	for k in range(1,T):
		i = random.randrange(S)
		step = 1/(k*lamda)
		if loss(w,data[i],labels[i])<1:
			w1 = ((1-step*lamda)*w)+step*data[i]*labels[i]
		elif loss(w,data[i],labels[i])>=1:
			w1 = ((1-step*lamda)*w)
		tt = (1/np.sqrt(lamda)/np.linalg.norm(w1))
		w1 = min(1,tt)*w1
		w = w1
	return w

		# if loss():



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
w=pegasos(X_train_binary,y_train_binary,6000)
correct=0
for k in range(0,len(y_test_binary)):
	if np.dot(w,X_test_binary[k])<0 and y_test_binary[k]<0:
		correct+=1
	elif np.dot(w,X_test_binary[k])>0 and y_test_binary[k]>0:
		correct+=1
#print(correct)
print("Test accuracy :",(correct*1.0/len(y_test_binary)))
