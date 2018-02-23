# Class to load faces and CIFAR-10 
import numpy as np
from PIL import Image
import time
from scipy import stats

class DataGenerator:
	def __init__(self, batch_size=150, faces_dir="",cifar_file="", train=0.9):
		cifar_X = self.load_cifar10(cifar_file)
		faces_X = self.load_faces(faces_dir)

		self.batch_size = batch_size
		self.X = np.concatenate((cifar_X,faces_X),0)
		self.X = self.X + np.random.sample(self.X.shape)*0.01 #Add small amount of noise to avoid /0 !!
		self.X = stats.zscore(self.X,0)
		self.Y = np.concatenate((np.repeat([[0,1]],len(cifar_X),0),np.repeat([[1,0]],len(faces_X),0)))

		

		indices = np.arange(self.X.shape[0])
		np.random.shuffle(indices)

		self.train = int(0.9*len(self.X))

		self.X_train = self.X[indices[:self.train]]
		self.Y_train = self.Y[indices[:self.train]]
		self.X_eval = self.X[indices[self.train:]]
		self.Y_eval = self.Y[indices[self.train:]]

	def batch(self):
		indices = np.arange(self.X_train.shape[0])
		np.random.shuffle(indices)

		while True:
			for i in range(0,len(indices),self.batch_size):
				yield self.X_train[i:i+self.batch_size,...],self.Y_train[i:i+self.batch_size]

	def eval(self):
		return self.X_eval, self.Y_eval

	def load_cifar10(self,file,number=400):
	    s = time.time()
	    import pickle
	    cifar = []
	    with open(file, 'rb') as fo:
	        dict_ = pickle.load(fo, encoding='bytes')
	        X = dict_[b'data'][:number,:].reshape((number,32,32,3), order='F')
	        for i in range(len(X)):
	        	img = Image.fromarray(X[i,:,:,:]).convert('1')
	        	cifar.append(np.array(img))
	    print('CIFAR loaded in: %fs'%(time.time()-s))
	    return np.asarray(cifar)

	def load_faces(self,faces_dir):
		import os
		s = time.time()
		faces = []
		for fd in os.listdir(faces_dir):
			for fi in os.listdir(faces_dir+"/"+fd):
				img = Image.open(faces_dir+"/"+fd+"/"+fi)
				img = np.array(img.convert('1').resize((32,32)))
				faces.append(img)

		print('Faces loaded in: %fs'%(time.time()-s))
		return np.asarray(faces)

