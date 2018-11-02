import numpy as np

N = 25
X_train = np.reshape(np.linspace(0, 0.9, N), (N, 1))

x = np.reshape(np.append(X_train[0:24], X_train[25:]), (N-1, 1))
print(x[0])
print(x.shape)


# colours = ['c-', 'g-', 'y-', 'r-', 'b-', 'm-']
# legend = []
# print(len(K_list))
# print(len(colours))
# for i in range(len(K_list)):
#     plt.plot(X_test, K_list[i], colours[i])
#     legend.append("K = " + str(i))
#
# plt.legend(legend, loc='lower left')
# plt.xlabel("X")
# plt.ylabel("Y")
# axes = plt.gca()
# axes.set_xlim([-1.0,1.2])
# axes.set_ylim([-1.3, 1.6])
# plt.show()
