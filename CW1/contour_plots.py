import numpy as np
import matplotlib.pyplot as plt
from answers import *

# Constants - Matrices
c = np.array([-1.0/6, -1.0/6])
C = np.array([[4, -1], [-1, 4]])
B = np.array([[3, -1], [-1, 3]])
a = np.array([1.0, 0])
b = np.array([0, -1.0])

def function2(x):
	x_minus_a = np.array([x[:,0] - a[0], x[:,1] - a[1]])

	sin = np.sin(np.dot(x_minus_a.T, x_minus_a))

	x_minus_b = np.array([x[:,0] - b[0], x[:,1] - b[1]])

	product_B = np.dot(x_minus_b.T, np.dot(B, x_minus_b))

	return sin + product_B

def function3(x):
	x_minus_a = np.array([x[:,0] - a[0], x[:,1] - a[1]])
	x_minus_b = np.array([x[:,0] - b[0], x[:,1] - b[1]])

	identity = np.identity(len(x)) / 100
	det = np.linalg.det(identity + np.dot(x, x.T))

	xTx = np.dot(x_minus_a.T, x_minus_a)
	xT_B_x = np.dot(x_minus_b.T, np.dot(B, x_minus_b))

	exp1 = np.exp(-xTx)
	exp2 = np.exp(-xT_B_x)
	log = np.log(det) / 10

	return 1 - (exp1 + exp2 - log)

X = np.arange(-1, 1, 0.1)
Y = np.arange(-1, 1, 0.1)
xx = np.column_stack((X,Y))

fig = plt.figure(figsize=plt.figaspect(0.5))

# Contour Plot of F1
ax = fig.add_subplot(1, 2, 1)

values2 = function2(xx)

steps2 = gradient_descent(0.1, 'f2')
print(steps2.shape)

ax.contour(X, Y, values2, 6, colors='k')

ax.plot(steps2[:,0], steps2[:,1], 'bo-')
# Contour Plot of F2
ax = fig.add_subplot(1, 2, 2)

values3 = function3(xx)

# steps3 = gradient_descent(0.1, 'f3')
# print(steps3.shape)
ax.contour(X, Y, values3, 6, colors='k')

plt.show()