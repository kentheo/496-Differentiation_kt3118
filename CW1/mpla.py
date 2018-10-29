import numpy as np
import matplotlib.pyplot as plt

# xlist = np.linspace(-2.0, 1.0, 100) # Create 1-D arrays for x,y dimensions
# ylist = np.linspace(-1.0, 2.0, 100)
# X,Y = np.meshgrid(xlist, ylist) # Create 2-D grid xlist,ylist values
# Z = np.sqrt(X**2 + Y**2) # Compute function values on the grid
# print(Z.shape)
# plt.contour(X, Y, Z, [0.5, 1.0, 1.2, 1.5], colors = 'k', linestyles = 'solid')
# plt.axes().set_aspect('equal') # Scale the plot size to get same aspect ratio
# plt.axis([-1.0, 1.0, -0.5, 0.5]) # Change axis limits
# plt.show()

def f(x):
    return (x[:,0]**2 + x[:,1]**2)

x = np.array([1,2,3])
y = np.array([1,2,3])
xx, yy = np.meshgrid(x, y)
X_grid = np.c_[ np.ravel(xx), np.ravel(yy) ]
print(X_grid)
# print(xx)
z = f(X_grid)

z = z.reshape(xx.shape)

plt.contour(xx, yy, z)
plt.show()