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
# pd.concat([dftrain, survived_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')
# plt.show()

# Encoding (adding numeric data) Categorical data (ex: sex "male", "female" --> 1, 2)
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []

for feature_name in CATEGORICAL_COLUMNS:
    vocab = dftrain[feature_name].unique() # gets a list of all unique values from given feature column (ex: ['male', 'female'])
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocab))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))


# Train the model: feed the model data
# Feed it in batches, according to the number of epochs (amt of times model will see same data)
# Input Function: defines how data will be broken into batches and epochs to be fed to model
def make_input_fn(data_df, label_df, num_epochs = 10, shuffle = True, batch_size = 32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))

        if shuffle:
            ds = ds.shuffle(1000)

        ds = ds.batch(batch_size).repeat(num_epochs)

        return ds

    return input_function

train_input_fn = make_input_fn(dftrain, survived_train)
eval_input_fn = make_input_fn(dfeval, survived_eval, num_epochs = 1, shuffle = False)

# Create the model:
linear_est = tf.estimator.LinearClassifier(feature_columns = feature_columns)

# Train the model:
linear_est.train(train_input_fn)

# Results from training
result = linear_est.evaluate(eval_input_fn)

clear_output() # clears console output
print(result['accuracy'])








# Some pandas learning (not related to the Titanic session):
# data = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])
# print(data)
# print()
# 
# data = {
#     'Name': ['Alice', 'Bob', 'Charlie'],
#     'Age': [25, 30, 35],
#     'Score': [85.5, 90.3, 95.1]
# }
# df = pd.DataFrame(data)
# # print(df[ ['Name', 'Score'] ])
# # print(df.iloc[0])
# print(df.loc[0])


