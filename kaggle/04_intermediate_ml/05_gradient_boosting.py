# Gradient Boosting: "Gradient Descent"
# - ensemble method (combines predictions of several methods, like random forest does with decision trees)

# How works:
# - Init ensemble with naive predictions
# - generate preds for initial ensemble
#   - Creates a loss function (how far off from target are we)
# - Use loss func to fit a new model that is added to ensemble
#   - Determine parameters (w,b) to reduce the loss
# - Add new model to ensemble

# Diagram:
# naive model -> make preds -> calc loss -> train new model -> add new model to ensemble -> ...
#             ...-> make preds (with new model)

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# Read the data
data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')

# Select features
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]

# Select target
y = data.Price

# Separate data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y)


# Use gradient boosting (gradient descent)
my_model = XGBRegressor(n_estimators=500)
my_model.fit(X_train, y_train)






# Parameter Tuning: Find best values for n_estimators

##############################
# n_estimators
##############################
# n_estimators: how many iterations of gradient descent to do (also # of models that we include in ensemble)
#  - too low: underfitting
#  - too high: overfitting
#  - typical: 100-1000


##############################
# early_stopping_rounds
##############################
# early_stopping_rounds: stop iterating when validation score stops improving
#  - good to set high n_estimators and use early_stopping_rounds
#  - early_stopping_rounds=5 good choice. Want to at least allow a few iterations if by chance the score doesn't start improving right away


##############################
# eval_set
##############################
# eval_set - set aside some data for calc'ing validation scores


# Figure out the best n_estimators to use, and then re-train model with all data:
my_model = XGBRegressor(n_estimators=500)
my_model.fit(X_train, y_train,
             early_stopping_rounds=5,
             eval_set=[(X_valid, y_valid)],
             verbose=False)

predictions = my_model.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))

# Once figure out best n_estimators, re-train model with n_estimators to be value you found most optimal with early stopping used above ^^

##############################
# Learning Rate
##############################
# Learning rate: each step (iteration) towards the target helps us less
# using higher n_estimators doesn't hurt us
# small learning rate (like 0.05) with large # of n_estimators is good approach. Default for learning rate is 0.1

# use learning rate:
my_model = XGBRegressor(n_estimators=500, learning_rate=0.05)
my_model.fit(X_train, y_train,
             early_stopping_rounds=5,
             eval_set=[(X_valid, y_valid)],
             verbose=False)


##############################
# n_jobs
##############################
# n_jobs: how many jobs to run in parallel: (good to set to amount of cores on your machine)
# only ideal for larger datasets
my_model = XGBRegressor(n_estimators=500, learning_rate=0.05, n_jobs=4)
my_model.fit(X_train, y_train,
             early_stopping_rounds=5,
             eval_set=[(X_valid, y_valid)],
             verbose=False)






