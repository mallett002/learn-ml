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
