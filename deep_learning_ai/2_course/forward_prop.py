import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Forward propagation in a single layer

x = np.array([200, 17])

# w2_1 = w, superscript 2, subscript 1 (2nd layer, first unit)

#### First Layer ####
# a1_1 = g(w1_1 • x + b1_1)

# First unit:
w1_1 = np.array([1, 2])
b1_1 = np.array([-1])
z1_1 = np.dot(w1_1, x) + b1_1
a1_1 = sigmoid(z1_1)

# Second unit:
w1_2 = np.array([-3, 4])
b1_2 = np.array([1])
z1_2 = np.dot(w1_2, x) + b1_2
a1_2 = sigmoid(z1_2)

# Third unit:
w1_3 = np.array([-3, 4])
b1_3 = np.array([1])
z1_3 = np.dot(w1_3, x) + b1_3
a1_3 = sigmoid(z1_3)

# Output first layer
a1 = np.array([a1_1, a1_2, a1_3])



#### Second Layer ####
# a2_1 = g(w2_1 • a1 + b2_1)

# only 1 unit:
w2_1 = np.array([-7, 8, 9])
b2_1 = np.array([3])
# note input (a1) is output from first layer:
z2_1 = np.dot(w2_1, a1) + b2_1
a2 = sigmoid(z2_1)

