# basic_sigmoid

import math
import numpy as np
def basic_sigmoid(x):

	s = 1.0/(1+1/math.exp())
	return s
basic_sigmoid(3)

#x = np.array([1,2,3])
def sigmoid(x):
	s= 1/0/(1+1/np.exp(x))
	return s

def sigmoid_derivative(x):
	s = 1.0/(1+1/np.exp(x))
	ds= s*(1-s)

	return ds