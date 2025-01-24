import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder


# How to deal with categorical vars??
# 3 approaches:
#   1 - Drop them             --> remove them (if not useful)
#   2 - ordinal encoding      --> assigning number
#       - Works well with "ordinal vars" (when there is a clear ranking relationship to vars)
#   3 - one hot endcoding      --> create cols, representing presence/absence of values
#       - works well when no clear ranking to vars (non ordinal) or "nominal vars"


#############################################################
# 1. LOAD SOME DATA
#############################################################

# 1. For training data, use only categorical features and numerical
#   Filter out features with range of values > 10
data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')

# Separate target from predictors
y = data.Price
X = data.drop(['Price'], axis=1)

# Divide data into training and validation subsets
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
# X_train_full: Feature training data for model
# X_valid_full: Feature validation of model
# y_train: Answer (target) for traning data
# y_valid: Answer (target) for validation of model

# Drop columns with missing values in training data (simplest approach)
cols_with_missing = [col for col in X_train_full.columns if X_train_full[col].isnull().any()] 
X_train_full.drop(cols_with_missing, axis=1, inplace=True)
X_valid_full.drop(cols_with_missing, axis=1, inplace=True)

# Build categorical data (Finding non-numerical data that only has a few, less than 10, unique values)
# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
    # if number of unique is less than 10 and it's an obj (string)
low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and
                        X_train_full[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
feature_cols = low_cardinality_cols + numerical_cols
X_train = X_train_full[feature_cols].copy()
X_valid = X_valid_full[feature_cols].copy()




# 2. Obtain a list of all of the categorical variables in the training data.
# create series of dtypes and if they are strings (index is colName)
s = (X_train.dtypes == 'object')
object_cols = list(s[s].index) # --> ['Type', 'Method', 'Regionname']




# 3. Test the different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

# Score from Approach 1 (Drop Categorical Variables)
drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_y_train = y_train.select_dtypes(exclude=['object'])
score_drop_vars = score_dataset(drop_X_train, drop_y_train, y_train, y_valid)
print(score_drop_vars)




# 4. Score from approach 2 (ordinal encoding)
# Make copy to avoid changing original data
label_X_train = X_train.copy()
label_X_valid = X_valid.copy()

# Apply ordinal encoder to each column with categorical data
ordinal_encoder = OrdinalEncoder()
label_X_train[object_cols] = ordinal_encoder.fit_transform(X_train[object_cols])
label_X_valid[object_cols] = ordinal_encoder.transform(X_valid[object_cols])

print("MAE from Approach 2 (Ordinal Encoding):") 
print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))




# 5. Score from Approach 3 (One-Hot Encoding)
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))

# one-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

# Add on-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

# Ensure all column names have type string 
OH_X_train.columns = OH_X_train.columns.astype(str)
OH_X_valid.columns = OH_X_valid.columns.astype(str)

print("MAE from Approach 3 (One-Hot Encoding):") 
print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))

# Typically One-Hot encoding performes the best
