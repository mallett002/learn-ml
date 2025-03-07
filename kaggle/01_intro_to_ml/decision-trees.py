import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

iowa_file_path = '../input/home-data-for-ml-course/train.csv'
home_data = pd.read_csv(iowa_file_path)

# By convention:
    # X is features
    # y is targets


# ---------------------------------------
# How to make a good model:
#   Training and Validation
#   Split the data into training data and validation data
#   Each has features (X) and targets (y)

#   Train them model

#   Make predictions with validation data 
#   Calculate MAE - See how far off from actual targets
#   Handle missing values
#    - drop cols with missing vals or imputation (fill them in with averages)
#    - see which one gives lower MAE

#   Find sweet spot with underfitting and overfitting
# ---------------------------------------



# -------------------------------------------------------------------------------------------------------------
# 1. Find prediction target (thing you want the AI to figure out)
home_data.columns # search through columns, looking for one related to home price

# Prediction target (predict home prices)
y = home_data.SalePrice


# -------------------------------------------------------------------------------------------------------------
# 2. Determine Features (inputs to train/fit the model)
feature_names = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
# Select data corresponding to features in feature_names
X = home_data[feature_names]

# Print statistics for X
print(X.describe())
# Print top few rows for featurs (X)
print(X.head())


# -------------------------------------------------------------------------------------------------------------
# 3. Split data into training and validation data
# train_X:  features for training
# train_y:  target for training 
# val_X:    features for validating how model is doing
# val_y:    target for validating how model is doing

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)


# -------------------------------------------------------------------------------------------------------------
# 4. Create the model
# For model reproducibility, set a numeric value for random_state when specifying the model
iowa_model = DecisionTreeRegressor(random_state=1)



# -------------------------------------------------------------------------------------------------------------
# 5. Fit (Train) the model
iowa_model.fit(train_X, train_y) # use training data (features X, and target y)



# -------------------------------------------------------------------------------------------------------------
# 6. Make predictions 
# Based on validation features, make predictions on what target (val_y) will be 
val_predictions = iowa_model.predict(val_X)

# print the top few validation predictions (targets that the model created)
print(val_predictions[:5])

# print the top few actual prices (expected prices) from validation data
print(val_y.head())

print("First in-sample predictions:", iowa_model.predict(X.head()))
print("Actual target values for those homes:", y.head().tolist())



# -------------------------------------------------------------------------------------------------------------
# 7. Calculate Mean Absolute Error
# Determines by how much are we off with the actual target
val_mae = mean_absolute_error(val_y, val_predictions) # --> 29652.931506849316

# Find sweet spot between underfitting and overfitting by comparing max_leaf_nodes' MAE
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    val_predictions = model.predict(val_X)
    mae = mean_absolute_error(val_y, val_predictions)
    return(mae)

# search throgh max_leaf_nodes to find the sweet spot btw underfitting and overfitting:
for max_leaf_nodes in [5, 25, 50, 100, 250, 500]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

# Don't split the data anymore now that we found our sweet spot btw underfitting/overfitting
final_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=0)

# fit the final model and uncomment the next two lines
final_model.fit(X, y)








