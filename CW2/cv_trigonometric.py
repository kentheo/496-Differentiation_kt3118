import numpy as np
import matplotlib.pyplot as plt

N = 25
X_train = np.reshape(np.linspace(0, 0.9, N), (N, 1))
Y_train = np.cos(10*X_train**2) + 0.1 * np.sin(100*X_train)

# Polynomial Design matrix of size N x (K+1)
def trigonometric_design_matrix(K, x):
    phi = np.zeros((len(x), (2*K)+1))
    for i in range(len(x)):
        phi[i][0] = 1.0
        for j in range(1, K+1):
            phi[i][(2*j)-1] = np.sin(2*np.pi*j*x[i])
            phi[i][2*j] = np.cos(2*np.pi*j*x[i])
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
    for i in range(len(x)):
        sum += (y[i] - np.dot(theta.T, phi[i]))**2
    return sum / len(x)

def gaussian_noise(mu, sigma, N):
    noise = np.zeros((N,1))
    # print(mu.shape)
    for i in range(N):
        noise[i] = np.random.normal(mu[i], sigma)
    # print(noise.shape)
    return noise

def get_mse_point(y_actual, y_received):
    return (y_actual - y_received)**2

def get_mse_full(errors):
    sum = 0
    for i in range(len(errors)):
        sum += errors[i]
    return sum / len(errors)

# Leave-one-out Cross Validation
# Returns a list of 2 elements
# Element 0: Average error
# Element 1: Sigma MLE
def linear_regression_trigonometric_CV(K, X_train, Y_train):
    errors = np.zeros((len(X_train), 1))
    sigma = np.zeros(())
    for i in range(len(X_train)):
        X_left_out = X_train[i]
        X_left_in = np.reshape(np.append(X_train[0:i], X_train[i+1:]), (len(X_train)-1, 1))
        Y_left_out = Y_train[i]
        Y_left_in = np.reshape(np.append(Y_train[0:i], Y_train[i+1:]), (len(Y_train)-1, 1))

        # phi_train = trigonometric_design_matrix(K, X_left_in)
        w = weights_MLE(K, X_left_in, Y_left_in)

        print("----------------- TEST DATA {0} iteration: {1}-------------------".format(K, i))
        phi_test = trigonometric_design_matrix(K, X_left_out)

        # This bit here is not necessary. DO NOT USE NOISE when finding the new Y
        # mu = np.dot(phi_test, w)
        # noise = gaussian_noise(mu, sigma, len(X_test))
        parametered_x = np.dot(phi_test, w)
        Y_test = parametered_x
        errors[i] = get_mse_point(Y_left_out, Y_test)
        print("------------------------------------------------")
    avg_error = get_mse_full(errors)
    print("Average error: {0}".format(avg_error))

    phi_train = trigonometric_design_matrix(K, X_train)
    w = weights_MLE(K, X_train, Y_train)
    sigma = sigma_MLE(w, phi_train, Y_train, X_train)

    return [avg_error[0], sigma]


# PLOTS
fig = plt.figure(figsize=plt.figaspect(0.5))

N_test = 200
X_test = np.reshape(np.linspace(-1.0, 1.2, N_test), (N_test, 1))

K_list = []
sigma_list = []
order_of_basis = []
for i in range(11):
    results = linear_regression_trigonometric_CV(i, X_train, Y_train)
    K_list.append(results[0])
    sigma_list.append(results[1])
    order_of_basis.append(i)

plt.plot(order_of_basis, K_list, 'r-', order_of_basis, sigma_list, 'b-')
plt.legend(["Average error", "Sigma MLE"], loc='upper right')
plt.xlabel("Order of basis")
plt.ylabel("Y")
plt.title("Average error and MLE for sigma for different orders of basis")

plt.show()
