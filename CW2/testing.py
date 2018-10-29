import numpy as np
import matplotlib.pyplot as plt

N = 25
X = np.reshape(np.linspace(0, 0.9, N), (N, 1))
Y = np.cos(10*X**2) + 0.1 * np.sin(100*X)

fig = plt.figure(figsize=plt.figaspect(0.5))

plt.plot(X, Y, 'ko')

plt.show()
