# homework:
# 1. replace sigmoid function with relu and tanh
# use google for definition and derivatives
# 2. play with learning rate value to converge neural network
# 3*. add one more layer

import numpy as np


# sigmoid
def f(x):
    return 1 / (1 + np.exp(-x))


# sigmoid derivative in terms of sigmoid values
def df(t):
    return t * (1 - t)


# dataset
# input - 3-dim array
# output - single number
# dataset size = 4
xs = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

ys = np.array([[0],
              [1],
              [1],
              [0]])

# random weights of hidden layer from -1 to 1
w1 = 2*np.random.random((3, 4)) - 1
# random weights of output layer from -1 to 1
w2 = 2*np.random.random((4, 1)) - 1


# predict y by x
def nn(x):
    a1 = f(np.dot(x, w1))
    a2 = f(np.dot(a1, w2))
    return a1, a2


def loss(l2_act):
    return np.sum(np.power(l2_act - ys, 2))


# learning rate
lr = 0.42


for j in xrange(60000):

    l0 = xs
    # l1 = f(np.dot(l0, w1))
    # l2 = f(np.dot(l1, w2))

    l1, l2 = nn(xs)

    l2_error = l2 - ys

    if (j % 1000) == 0:
        print "Error: " + str(np.mean(np.abs(l2_error)))

    l2_delta = l2_error * df(l2)

    l1_error = l2_delta.dot(w2.T)

    l1_delta = l1_error * df(l1)

    w2 -= lr * l1.T.dot(l2_delta)
    w1 -= lr * l0.T.dot(l1_delta)

print('true Y values:')
print(ys)
print('predicted values:')
print(l2)
