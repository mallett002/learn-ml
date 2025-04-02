import numpy as np


A = np.array([ [1, -1, 0.1],
               [2, -2, 0.2]])

# Transpoosese of A:
AT = A.T

W = np.array([[3, 5, 7, 9],
              [4, 6, 8, 0]])

# matrix multiplication 
Z = np.matmul(AT, W)

# Can also just use "@" operator for matrix multiplication:
Z2 = AT @ W

# Transpose of W:
# print(f"A: {A}")
# print("\n")
# print(f"AT: {AT}")
print(f"Z: {Z}")
print("\n")
print(f"Z2: {Z2}")


# Forward Prop in NN: *****************************************
# Dense layer vectorized:
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Transposed: rows
# W columns
AT = np.array([[200, 17]])
W = np.array([[1, -3, 5],
              [2, 4, -6]])
b = np.array([[-1, 1, 2]])

def dense(AT, W, b):
    z = np.matmul(AT, W) + b
    return sigmoid(z) 

a_out = dense(AT, W, b) # [[1, 0, 1]]
