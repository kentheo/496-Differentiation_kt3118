import numpy as np
import matplotlib.pyplot as plt

N = 25
X_train = np.reshape(np.linspace(0, 0.9, N), (N, 1))
Y_train = np.cos(10*X_train**2) + 0.1 * np.sin(100*X_train)

# Polynomial Design matrix of size N x (K+1)
def trigonometric_design_matrix(K, x):
    phi = np.zeros((len(x), (2*K)+1))
    print("phi shape = {0}".format(phi.shape))
    for i in range(len(x)):
        phi[i][0] = 1.0
        for j in range(1, K+1):
            print("j = {0}, i = {1}".format((2*j)-1, i))
            phi[i][(2*j)-1] = np.sin(2*np.pi*j*x[i])
            phi[i][2*j] = np.cos(2*np.pi*j*x[i])
    print(phi)
    return phi

# Estimate weights MLE
def weights_MLE(K, X, Y):
    phi_k = trigonometric_design_matrix(K, X)
    try:
        inverse = np.linalg.inv(np.dot(phi_k.T, phi_k))
    except np.linalg.LinAlgError:
        print("ERROR!! Matrix not invertible")
        pass
    else:
        temp = np.dot(inverse, phi_k.T)
        w_MLE = np.dot(temp, Y)
        return w_MLE
    # SHOULD NEVER GET HERE
    print("ERROR!!!!! Can't return w_MLE")
    return phi_k

# Estimate sigma MLE
def sigma_MLE(theta, phi, y, x):
    sum = 0
    print("pfffff shape = {0}".format(phi.shape))
    for i in range(len(x)):
        sum += (y[i] - np.dot(theta.T, phi[i]))**2
    print("sum = {0}".format(sum))
    return sum / len(x)

def gaussian_noise(mu, sigma, N):
    noise = np.zeros((N,1))
    # print(mu.shape)
    for i in range(N):
        noise[i] = np.random.normal(mu[i], sigma)
    # print(noise.shape)
    return noise

def linear_regression_trigonometric(K, X_train, Y_train, X_test):
    phi_train = trigonometric_design_matrix(K, X_train)
    w = weights_MLE(K, X_train, Y_train)
    print("w shape = {0}".format(w.shape))
    sigma = sigma_MLE(w, phi_train, Y_train, X_train)
    print("sigma shape = {0}".format(sigma.shape))

    print("----------------- TEST DATA {0} -------------------".format(K))
    phi_test = trigonometric_design_matrix(K, X_test)

    # This bit here is not necessary. DO NOT USE NOISE when finding the new Y
    # mu = np.dot(phi_test, w)
    # noise = gaussian_noise(mu, sigma, len(X_test))
    parametered_x = np.dot(phi_test, w)
    Y_test = parametered_x
    print("------------------------------------------------")
    return Y_test

fig = plt.figure(figsize=plt.figaspect(0.5))

N_test = 200
X_test = np.reshape(np.linspace(-1.0, 1.2, N_test), (N_test, 1))

# Linear regrssion for each trigonometric value
K_1 = linear_regression_trigonometric(1, X_train, Y_train, X_test)
K_11 = linear_regression_trigonometric(11, X_train, Y_train, X_test)

plt.plot(X_test, K_1, 'g-', X_test, K_11, 'r-')
plt.legend(["K = 1", "K = 11"], loc='lower left')
plt.xlabel("X")
plt.ylabel("Y")
axes = plt.gca()
axes.set_xlim([-1.0,1.2])
axes.set_ylim([-1.3, 1.6])
plt.show()
