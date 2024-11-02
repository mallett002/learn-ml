import matplotlib.pyplot as plt # visualization of graphs 
import numpy as np # optimized version of arrays in python
import pandas as pd # data analytics and manipulation tool
import tensorflow as tf

from tensorflow import feature_column as fc
from IPython.display import clear_output
from six.moves import urllib


# Linear regression - Creates a line of best fit
#   - datapoints that are in a linear fashion. for prediction

# x = [1, 2, 2.5, 3, 4]
# y = [1, 4, 7, 9, 15]

# plt.plot(x, y, 'ro')
# plt.axis([0, 6, 0, 20])
# plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
# plt.show()

# Testing out likelyhood of survival on titanic:
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data

# print('.head(): First 5 items in dftrain:')
# print(dftrain.head())

# remove survived from data frames (This is what we're trying to solve)
survived_train = dftrain.pop('survived')
survived_eval = dfeval.pop('survived')

# print('.head(): First 5 items in dftrain:')
# print(dftrain.head())

# print('analysis:')
# print(dftrain.describe())

# print('shape:')
# print(dftrain.shape)

# print('survived_train:')
# print(survived_train)

# dftrain.age.hist(bins=20)
# dftrain.sex.value_counts().plot(kind='barh')
# dftrain['class'].value_counts().plot(kind='barh')
# group by sex, average survival per sex
pd.concat([dftrain, survived_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')
plt.show()