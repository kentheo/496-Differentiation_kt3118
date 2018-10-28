# Script to test grad functions

import numpy as np
from answers import *

c = np.array([1.0/6, 1.0/6])
C = np.array([[4, -1], [-1, 4]])
B = np.array([[3, -1], [-1, 3]])
a = np.array([1.0, 0])
b = np.array([0, -1.0])

xx_minus_c = (a + b) / 2

C_inverse = np.linalg.inv(C)

c0 = np.dot(-c,np.dot(C,c))
# print 'c0 = {0}'.format(c0)

x = np.array([1.0/6, 1.0/6])

def test_grad(x, grad_function):
	grad = grad_function(x)
	print 'x = {0} of shape {1}'.format(x, x.shape)
	print 'xT = {0} of shape {1}'.format(x.T, x.T.shape)
	print 'grad_function = {0} of shape {1}'.format(grad, grad.shape)

def function1(x):
	# (x[:,0]**2 + x[:,1]**2)
	x_minus_c = x-c

	product_C = np.dot(x_minus_c.T, np.dot(C, x_minus_c))

	return product_C - (1.0/6)


test_grad(c, grad_f1)

test_grad(np.array([0, 0]), grad_f2)

test_grad(np.array([0.0, 1.0]), grad_f3)

print(function1(x))