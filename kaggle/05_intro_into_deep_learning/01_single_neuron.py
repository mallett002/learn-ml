import pandas as pd

from tensorflow import keras
from tensorflow.keras import layers

# The Linear Unit:  y = wx + b
# A neural network learns by modifying its weights
# y = prediction (y^)
# w is the weight 
# x is input 
# b is bias (Base value when x = 0)

# So when you calculate the prediction 
# given input x (sugars per serving) calc y^ amt of calories per serving
# Might find: Bias:90, w=2.5
# For 5 grams of sugar, `𝑦 = 2.5 × 5 + 90 = 102.5`
#  calories. This means the cereal is estimated to have 102.5 calories per serving.

# More inputs
# y = w1x1 + w2x2 + w3x3 + b


#Linear Units in Keras
# layer with 3 inputs and 1 neuron ( ouput unit )
model = keras.Sequential([
    layers.Dense(units=1, input_shape=[3])
])



# Exercise - A Single Neuron
red_wine = pd.read_csv('data/red-wine.csv')
red_wine.head()
red_wine.shape # ( 1599, 12 ) # 1599 rows, 12 columns

model = keras.Sequential([
    layers.Dense(units=1, input_shape=[11])
])

# see the weights and bias
w, b = model.weights
print("Weights\n{}\n\nBias\n{}".format(w,b))


