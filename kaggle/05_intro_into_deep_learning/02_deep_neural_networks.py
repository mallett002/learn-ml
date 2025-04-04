from tensorflow import keras
from tensorflow.keras import layers
# from tensorflow import keras
# from keras import layers

# Activation functions: function applied to the output of a neuron
# ex: max(0,x)

# # ReLU (Rectified Linear Unit) activation function
# activation output = max(0, wx + b)

# Sequential model
model = keras.Sequential([
    # the hidden ReLU layers
    layers.Dense(units=4, activation='relu', input_shape=[2]),
    layers.Dense(units=3, activation='relu'),

    # the linear output layer 
    layers.Dense(units=1),
])




# Deep Neural Networks exercise ##################################
model = keras.Sequential([
    layers.Dense(units=512, activation='relu', input_shape=[8]),
    layers.Dense(units=512, activation='relu'),
    layers.Dense(units=512, activation='relu'),
    layers.Dense(units=1),
])

# Can put some other layer in between a layer and its activation function:
# (Equivalent to model above)
model = keras.Sequential([
    layers.Dense(512, input_shape=[8]),
    layers.Activation('relu'),

    layers.Dense(512),
    layers.Activation('relu'),

    layers.Dense(512),
    layers.Activation('relu'),

    layers.Dense(1),
])

# Many variations of the ReLU activation function:
# LeakyReLU: allows a small gradient when the input is negative
# Exponential Linear Unit (ELU): smooths the transition from negative to positive
# Scaled Exponential Linear Unit (SELU): self-normalizing property
# Swish: smooth transition from negative to positive
# Softplus: smooth approximation of the ReLU function
# Softmax: used in multi-class classification problems
# Sigmoid: used in binary classification problems
# Tanh: used in binary classification problems

