
# coding: utf-8
"""
Use this file for your answers.

This file should been in the root of the repository
(do not move it or change the file name)

"""

# NB this is tested on python 2.7. Watch out for integer division

import numpy as np

def grad_f1(x):
    """
    4 marks

    :param x: input array with shape (2, )
    :return: the gradient of f1, with shape (2, )
    """
    C = np.array([[4, -1], [-1, 4]])
    c = np.array([1.0/6, 1.0/6])

    grad = 2 * np.dot((x-c).T, C)
    return grad

def grad_f2(x):
    """
    6 marks

    :param x: input array with shape (2, )
    :return: the gradient of f2, with shape (2, )
    """
    a = np.array([1, 0])
    b = np.array([0, -1])
    B = np.array([[3, -1], [-1,3]])

    x_minus_a = x - a
    x_minus_b = x - b

    grad = (2 * np.dot(np.cos(np.dot(x_minus_a.T, x_minus_a)), x_minus_a.T)) + (2 * np.dot(x_minus_b.T, B))

    return grad

def grad_f3(x):
    """
    This question is optional. The test will still run (so you can see if you are correct by
    looking at the testResults.txt file), but the marks are for grad_f1 and grad_f2 only.

    Do not delete this function.

    :param x: input array with shape (2, )
    :return: the gradient of f3, with shape (2, )
    """
    a = np.array([1, 0])
    b = np.array([0, -1])
    B = np.array([[3, -1], [-1,3]])

    x_minus_a = x - a
    x_minus_b = x - b

    exp_1 = np.dot(x_minus_a.T, x_minus_a)
    exp_2 = np.dot(x_minus_b.T, np.dot(B, x_minus_b))
    
    x_T = np.array([x]).T

    id_x = 1/100.0* np.identity(2) + (x* x_T)
    
    determinant = np.linalg.det(id_x)
    
    grad = np.exp(-exp_1)*(2 * x.T) + np.exp(-exp_2)*(2 * np.dot(x.T, B)) + (((2/1000) * x.T) / determinant)
    
    return grad