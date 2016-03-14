#coding:utf-8
import numpy as np
import pylab
from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
from sklearn import cross_validation



class perceptron:

	# d:num of feature k:number of class
	def __init__(self, d,k,iter_num):

		# w = [[d]*k] array
		self.w = np.random.random(d*k).reshape(k, d)
		self.iter_num = iter_num



	#こことかあんま自信ない<- きたない
	def infer(self,x_data,y_data):
		predicts = np.argmax(np.dot(x_data, self.w.T), axis=1)
		accuracy = len(filter(lambda x: x, predicts == y_data)) / float(len(y_data))
		return accuracy


	def train(self,x_data,y_data):

		#iter_num回 training
		for n in range(self.iter_num):

			for x,y in zip(x_data, y_data):

				i = np.argmax(np.dot(x, self.w.T))

				if y != i:
					self.w[i] -= float(self.iter_num - i) / self.iter_num * x
					self.w[y] += float(self.iter_num - i) / self.iter_num * x

			a = self.infer(x_data, y_data)
			print "iter%d:train accuracy:%f" % (n, a)



if __name__ == "__main__":
	mnist = fetch_mldata('MNIST original', data_home=".")
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(mnist.data, mnist.target, test_size=0.4, random_state=0)
	p = perceptron(784, 10, 100)
	p.train(X_train, y_train)
	print p.infer(X_test, y_test)