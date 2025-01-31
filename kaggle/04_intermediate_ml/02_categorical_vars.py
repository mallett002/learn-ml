import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder


# How to deal with categorical vars??
# Get errors trying to plug them into ML models w/o pre-processing them first
# 3 approaches:
#   1 - Drop them             --> remove them (if not useful)
#   2 - Ordinal encoding      --> assigning number
#       - Works well with "ordinal vars" (when there is a clear ranking relationship to vars)
#   3 - One Hot Endcoding      --> create cols, representing presence/absence of values
#       - Works well when no clear ranking to vars (non ordinal) or "nominal vars"


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

# Drop columns with missing values in training data (to keep things simple for setting up data for all 3 approaches)
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
X_categ_num_train = X_train_full[feature_cols].copy()
X_categ_num_valid = X_valid_full[feature_cols].copy()




# 2. Obtain a list of all of the categorical variables in the training data.
# create series of dtypes and if they are strings (index is colName)
s = (X_categ_num_train.dtypes == 'object')
categorical_cols = list(s[s].index) # --> ['Type', 'Method', 'Regionname']




# 3. Test the different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

# Score from Approach 1 (Drop Categorical Variables)
drop_X_train = X_categ_num_train.select_dtypes(exclude=['object'])
drop_y_train = y_train.select_dtypes(exclude=['object'])
score_drop_vars = score_dataset(drop_X_train, drop_y_train, y_train, y_valid)
print(score_drop_vars)




# 4. Score from approach 2 (ordinal encoding)
# Make copy to avoid changing original data
label_X_train = X_categ_num_train.copy()
label_X_valid = X_categ_num_valid.copy()

# Apply ordinal encoder to each column with categorical data
ordinal_encoder = OrdinalEncoder()
label_X_train[categorical_cols] = ordinal_encoder.fit_transform(X_categ_num_train[categorical_cols])
label_X_valid[categorical_cols] = ordinal_encoder.transform(X_categ_num_valid[categorical_cols])

print("MAE from Approach 2 (Ordinal Encoding):") 
print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))




# 5. Score from Approach 3 (One-Hot Encoding)
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
# handle_unknown='ignore': avoid errors if validation data has classes different from training set
# sparse=False: return encoded cols as numpy array instead of sparse matrix

# Do the encodings and save them to a new dataframe
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_categ_num_train[categorical_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_categ_num_valid[categorical_cols]))

# one-hot encoding removed index; put it back
OH_cols_train.index = X_categ_num_train.index
OH_cols_valid.index = X_categ_num_valid.index

# Remove categorical columns from (will replace with one-hot encoding)
num_X_train = X_categ_num_train.drop(categorical_cols, axis=1)
num_X_valid = X_categ_num_valid.drop(categorical_cols, axis=1)

# Combine hot encoded cols df with numerical cols df
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

# Ensure all column names have type string 
OH_X_train.columns = OH_X_train.columns.astype(str)
OH_X_valid.columns = OH_X_valid.columns.astype(str)

print("MAE from Approach 3 (One-Hot Encoding):") 
print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))

# Typically One-Hot encoding performes the best











#############################################################
# CATEGORICAL VARS EXERCISE
#############################################################

# 1. load data and test split it
X = pd.read_csv('../input/train.csv', index_col='Id') 
X_test = pd.read_csv('../input/test.csv', index_col='Id')

# remove rows with missing target, separate target from predictors
X.dropna(axis=0, subset=['SalePrice'], inplace=True,) # 0: drop row, subset: only look at SalePrice, inplace: dont create whole new DF

# target
y = X.SalePrice

# Remove target col from training data
X.drop(['SalePrice'], axis=1, inplace=True)

# to keep things simple, drop cols w/ missing values
cols_with_missing = [col for col in X.columns if X[col].isnull().any()]
X.drop(cols_with_missing, axis=1, inplace=True)
X_test.drop(cols_with_missing, axis=1, inplace=True)

# Separate validation set from training set
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
# random_state=0: ensures you always get the same test split of the data for reproducibility
# X_train: Features for training
# X_valid: Freatures for validation
# y_train: Target for training
# y_valid: Target for validation

# Compare different models, def score_dataset func
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)

    return mean_absolute_error(y_valid, preds) # See how far off y-hat (preds) is from actual (y_valid)


# 2. Compare the different approaches

# 2.a Drop columns approach
# Remove object (string) types
drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])
print("MAE from Approach 1 (Drop categorical variables):")
score_dataset(drop_X_train, drop_X_valid, y_train, y_valid) # 17837.82570776256



# 2.b Ordinal Encoder:
# Note, the training data and the validation data have different unique values for a column.
# this will cause an error
print("Unique values in 'Condition2' column in training data:", X_train['Condition2'].unique())
print("\nUnique values in 'Condition2' column in validation data:", X_valid['Condition2'].unique())
# Unique values in 'Condition2' column in training data: ['Norm' 'PosA' 'Feedr' 'PosN' 'Artery' 'RRAe']
# Unique values in 'Condition2' column in validation data: ['Norm' 'RRAn' 'RRNn' 'Artery' 'Feedr' 'PosN']

# Can write custom ordinal encoder to deal with new categories.
# Can also just drop problematic columns (easier) as seen here:

# get the cols with values of type object (string)
object_cols = [col for col in X_train.columns if X_train[col].dtype == 'object']

# get columns with values present in both testing data and training data
good_label_cols = [col for col in object_cols if
                   set(X_valid[col]).issubset(set(X_train[col]))]

# Remove good from all to leave just the bad ones
bad_label_cols = list(set(object_cols)-set(good_label_cols))

# Drop categorical columns that will not be encoded
# Create new DFs with bad cols removed
label_X_train = X_train.drop(bad_label_cols, axis=1)
label_X_valid = X_valid.drop(bad_label_cols, axis=1)

# Apply Ordinal Encoder
ordinal_encoder = OrdinalEncoder()
# encode the good cols in the data, and save them back to the new dfs (label_X_...)
label_X_train[good_label_cols] = ordinal_encoder.fit_transform(X_train[good_label_cols])
label_X_valid[good_label_cols] = ordinal_encoder.transform(X_valid[good_label_cols])


print("MAE from Approach 2 (Ordinal Encoding):") 
print(score_dataset(label_X_train, label_X_valid, y_train, y_valid)) # 17098.01649543379




# 2.c. Investigate Cardinality
# Get number of unique entries in each column with categorical data
object_nunique = list(map(lambda col: X_train[col].nunique(), object_cols)) # [5, 2, 4, 4, ...]
pairs = zip(object_cols, object_nunique) # ('Location', 5) ('HasGarage', 2) ...
d = dict(pairs) # turn pairs into nice dictionary 

# Print number of unique entries by column, in ascending order
sorted(d.items(), key=lambda x: x[1]) # sort by the nunique vals [('Street', 2), ('LandSlope', 3), ('LotShape', 4), ...]






# 2.d. One hot encoding:
# 'Neighborhood' has cardinality of 25 (25 options)
# can make dataset huge if we one hot encode all of these columns (especially ones with high cardinality)
# only one-hot-encode low cardinality values - drop others or use ordinal encoding

