import numpy as np
import matplotlib.pyplot as plt

N = 25
X_train = np.reshape(np.linspace(0, 0.9, N), (N, 1))
Y_train = np.cos(10*X_train**2) + 0.1 * np.sin(100*X_train)

# Gaussian Design matrix of size N x (K+1)
def gaussian_design_matrix(K, x, l):
    phi = np.zeros((len(x), K+1))
    # Get means of m_i
    means = np.reshape(np.linspace(0, 1.0, K+1), (K+1, 1))
    for i in range(len(x)):
        phi[i][0] = 1.0
        for j in range(1, K+1):
            term = (x[i] - means[j])**2
            term /= (2*l)**2
            phi[i][j] = np.exp(-term)
    print("phi shape = {0}".format(phi.shape))
    print(phi)
    return phi

# Estimate weights MAP
def weights_MAP(K, X, Y, l, phi_k, lamda):
    phi_k_T_phi = np.dot(phi_k.T, phi_k)
    b_squared =  1 / (2 * lamda)
    # sigma = (2*l)**2
    term = (l / b_squared) * np.identity(len(phi_k_T_phi))
    try:
        inverse = np.linalg.inv(phi_k_T_phi + term)
    except np.linalg.LinAlgError:
        print("ERROR!! Matrix not invertible")
        pass
    else:
        temp = np.dot(inverse, phi_k.T)
        w_MAP = np.dot(temp, Y)
        return w_MAP
    # SHOULD NEVER GET HERE
    print("ERROR!!!!! Can't return w_MLE")
    return phi_k

def linear_regression_gaussian(K, X_train, Y_train, X_test, lamda):
    l = 0.1
    phi_train = gaussian_design_matrix(K, X_train, l)
    w = weights_MAP(K, X_train, Y_train, l, phi_train, lamda)
    print("w shape = {0}".format(w.shape))

    print("----------------- TEST DATA {0} -------------------".format(K))
    phi_test = gaussian_design_matrix(K, X_test, l)

    parametered_x = np.dot(phi_test, w)

    print("parametered_x shape = {0}".format(parametered_x.shape))
    Y_test = parametered_x
    print("------------------------------------------------")
    return Y_test

fig = plt.figure(1,figsize=(11,9))

N_test = 200
X_test = np.reshape(np.linspace(-0.3, 1.3, N_test), (N_test, 1))

# Linear regrssion for each gaussian value
K_20 = linear_regression_gaussian(20, X_train, Y_train, X_test, 0.1)

plt.plot(X_train, Y_train, 'ko', X_test, K_20, 'b-')
plt.legend(["Original", "K = 20"], loc='lower left')
plt.xlabel("X")
plt.ylabel("Y")
axes = plt.gca()
axes.set_xlim([-0.3,1.3])
# axes.set_ylim([-1.4,3.6])
fig.show()

fig2 = plt.figure(2,figsize=(11,9))
K_20_2 = linear_regression_gaussian(20, X_train, Y_train, X_test, 20.0)

plt.plot(X_train, Y_train, 'ko', X_test, K_20_2, 'b-')
plt.legend(["Original", "K = 20"], loc='lower left')
plt.xlabel("X")
plt.ylabel("Y")
axes = plt.gca()
axes.set_xlim([-0.3,1.3])
# axes.set_ylim([-1.4,3.6])
fig2.show()

fig3 = plt.figure(3,figsize=(11,9))
K_20_3 = linear_regression_gaussian(20, X_train, Y_train, X_test, 0.001)

plt.plot(X_train, Y_train, 'ko', X_test, K_20_3, 'b-')
plt.legend(["Original", "K = 20"], loc='lower left')
plt.xlabel("X")
plt.ylabel("Y")
axes = plt.gca()
axes.set_xlim([-0.3,1.3])
# axes.set_ylim([-1.4,3.6])
fig3.show()
raw_input()
