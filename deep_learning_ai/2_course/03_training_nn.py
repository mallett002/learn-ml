import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense 
from tensorflow.keras.losses import BinaryCrossentropy 

# 1. Create the neural network model
nn_model = Sequential([
    Dense(units=25, activation='sigmoid'),
    Dense(units=15, activation='sigmoid'),
    Dense(units=1, activation='sigmoid'),
])

# 2. Compile the model
nn_model.compile(loss=BinaryCrossentropy())

# Create some dummy data
X = np.random.randn(100, 10)
y = np.random.randint(0, 2, size=(100, 1))

# 3. Train the model
nn_model.fit(X, y, epochs=10)
# Epochs: number of steps for gradient descent

