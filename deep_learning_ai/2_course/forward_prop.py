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



#######################################
### More common/programatic approach ##
#######################################

# define a dense func:
def dense(a_in, W, b):
    units = W.shape[1] # Get the amount of units (3 in this ex)
    a_out = np.zeros(units) # start as the amount of units (the amount of outputs)

    for j in range(units): # 0, 1, 2
        w = W[:, j] # pull out "j"th col in the matrix (w1_1, then w1_2, etc.)
        z = np.dot(w, a_in) + b[j]
        a_out[j] = sigmoid(z)
    
    return a_out

# How you would define sequential to create the Nueral network:
# def sequential(X):
#     a1 = dense(X, W1, b1) # note: capitals refering to matrixes in linear algebra
#     a2 = dense(a1, W2, b2)
#     a3 = dense(a2, W3, b3)
#     a4 = dense(a3, W4, b4)
#     f_x = a4
#     return f_x


# W for first layer:
# w1_1 = [1,2]
# w1_2 = [-3,4]
# w1_3 = [5,-6]

# 2 x 3 array
# 2 rows: number of input features
# 3 cols: number of nuerons (units)
W = np.array([
    [1, -3, 5], # feature set 1
    [2, 4, -6]  # feature set 2
]) # ^  ^   ^  
#   n1  n2  n3


# b for first layer:
# b1_1 = -1
# b1_2 = 1
# b1_3 = 2
b = np.array([-1, 1, 2])

# X or a[0]
a_in = np.array([-2, 4])
