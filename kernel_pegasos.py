import numpy as np
import random, math, sys, os
from tqdm import tqdm

def load_mnist(path, kind='train'):
	#load data from given ubyte format
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

def kernel_f(k,v1,v2,degree):
	#implementation of kernels
	if k == 'homogenious':
		op = np.dot(v1, v2)
		op=np.power(op,degree)
		return op
	elif k == 'radial':
		v=v1-v2
		op = math.exp(-1 * degree * np.linalg.norm(v) ** 2)
		return op

def loss(w,x,y):
	#loss for linear SVM
	op = y*np.dot(w,x)
	return op
	
def mercer_pegasos(kernel,data,labels,T, deg, lamda=1.0):
	if kernel=='linear':
		#Linear implementation of Pegasos SVM
		print("Linear Kernel with {} iterations".format(T))
		S = len(data)
		w = np.zeros((data[0].shape[0]))
		for k in tqdm(range(1,T)):
			i = random.randrange(S)
			step = 1/(k*lamda)
			if loss(w,data[i],labels[i])<1:
				w1 = ((1-step*lamda)*w)+step*data[i]*labels[i]
			elif loss(w,data[i],labels[i])>=1:
				w1 = ((1-step*lamda)*w)
			tt = min(1,((1/np.sqrt(lamda)/np.linalg.norm(w1))))*w1 #weight update per each sample
			w=tt
		return tt
	else:
		print("Kernel SVM, with {} kernel, {} degree, {} iterations. Learning alpha's, please wait....".format(kernel,deg,T))
		#Kernel SVM implemetation
		S=len(data)
		alpha = np.zeros((S))
		w = np.zeros((data[0].shape[0]))
		for k in tqdm(range(1,T)):
			it = random.randrange(S)
			sm=0
			for j in range(0,S):
				sm+=alpha[j]*labels[it]*kernel_f(kernel,data[it],data[j],deg)
			if labels[it]*(1/(k*lamda))*sm <1:
				alpha[it] = alpha[it]+1 #Update date of alpha vaules

		for k in range(0,S):
			w += alpha[k]*labels[k]*data[k] #Converting alpha's to weights for inference
		return w

def get_data(path,binary):
	#conversion of data from multi labels to -1 and +1 labels for two classes (1 and 2)
	X_train, y_train = load_mnist(path, kind='train')
	X_test, y_test = load_mnist(path, kind='t10k')
	if not binary:
		return X_train,y_train,X_test,y_test
	else:
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
		return X_train_binary,y_train_binary,X_test_binary,y_test_binary

def test_acc(X_test_binary,y_test_binary,w):
	correct=0
	for k in range(0,len(y_test_binary)):
		if np.dot(w,X_test_binary[k])<0 and y_test_binary[k]<0:
			correct+=1
		elif np.dot(w,X_test_binary[k])>0 and y_test_binary[k]>0:
			correct+=1
	return (correct*1.0/len(y_test_binary))

def run_expts(arg):
	X_train_binary,y_train_binary,X_test_binary,y_test_binary = get_data('fashionmnist/',True)
	kernels_list = ['radial','homogenious']
	degree_list = [2,3,4,5]
	iter_list = [500,1000,2000,3000,4000,5000]
	linear_itr_list = [500,1000,2000,3000,5000,7000,10000]
	if arg=='all':
		for itr in linear_itr_list:
			w = mercer_pegasos('linear',X_train_binary,y_train_binary, itr, 2)
			print("test acc :",test_acc(X_test_binary,y_test_binary,w))
		sys.exit()
		for krnl in kernels_list:
			for dgree in degree_list:
				for itr in iter_list:
					w = mercer_pegasos(krnl,X_train_binary,y_train_binary, itr, dgree)
					print("test acc :",test_acc(X_test_binary,y_test_binary,w))
	elif arg=='linear':
		w = mercer_pegasos('linear',X_train_binary,y_train_binary, 6000, 2)
		print("test acc :",test_acc(X_test_binary,y_test_binary,w))
	elif arg=='radial':
		w = mercer_pegasos('radial',X_train_binary,y_train_binary, 500, 2)
		print("test acc :",test_acc(X_test_binary,y_test_binary,w))
	elif arg=='homogenious':
		w = mercer_pegasos('homogenious',X_train_binary,y_train_binary, 500, 2)
		print("test acc :",test_acc(X_test_binary,y_test_binary,w))

# run_expts('linear') #Runs Pegasos linear SVM
# run_expts('radial') #Runs Kernelized SVM with radial basis as kernel
# run_expts('homogenious') #Runs kernelized SVM with homogenious polynomial as kernel
run_expts('all') #Runs all possible combinations of kernels, parameters and iterations for analysis
