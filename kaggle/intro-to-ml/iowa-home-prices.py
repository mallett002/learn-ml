import pandas as pd

from sklearn.tree import DecisionTreeRegressor

home_data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv') 

#-------------------------------------------------------------------------------------------------
# 1. Find prediction target 
#-------------------------------------------------------------------------------------------------
home_data.columns

# Prediction target
y = home_data.SalePrice


#-------------------------------------------------------------------------------------------------
# 2. Determine Features (inputs to train/fit the model)
#-------------------------------------------------------------------------------------------------
feature_names = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
# Select data corresponding to features in feature_names
X = home_data[feature_names]

# Print statistics for X
print(X.describe())

# Print top few rows for featurs (X)
print(X.head())


#-------------------------------------------------------------------------------------------------
# 3. Create the model
#-------------------------------------------------------------------------------------------------
# For model reproducibility, set a numeric value for random_state when specifying the model
iowa_model = DecisionTreeRegressor(random_state=1)


#-------------------------------------------------------------------------------------------------
# 4. Fit the model
#-------------------------------------------------------------------------------------------------
iowa_model.fit(X, y)


#-------------------------------------------------------------------------------------------------
# 5. Make predictions 
#   In practice, you'll want to make predictions for new houses coming on the market rather than
#   the houses we already have prices for.
#-------------------------------------------------------------------------------------------------
# create predictions with predict()
predictions = iowa_model.predict(X)

# Create comparison dataframe with predictions and actual values
comparison = pd.DataFrame({'Predictions': predictions, 'Actual Values': y})

# Display the first few rows
print(comparison.head())




