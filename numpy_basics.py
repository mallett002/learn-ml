import numpy as np

arr = np.array([1, 2, 3])
zeros = np.zeros((3, 3))
ones = np.ones((3, 3))
zero_to_nine = np.arange(10)



a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# operations based on index
sum_arr = a + b # --> [5 7 9]
product_arr = a * b # --> [4 10 18]

print(product_arr)


# Methods
nums = np.array([1, 2, 3, 4, 5])

# Calculate the sum of elements
print(nums.sum())

# Calculate the mean of elements
print(nums.mean())

# Find the index of the maximum value
print(nums.argmax())



# Create a 2D array (matrix)
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(matrix)

# Accessing an element in a 2D array
print(matrix[1, 2])

# Slicing a 2D array
print(matrix[:, 1])

