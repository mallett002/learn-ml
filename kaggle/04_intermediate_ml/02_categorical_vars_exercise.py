
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder


#############################################################
# CATEGORICAL VARS EXERCISE
#############################################################


#############################################################
# 1. load data and test split it
#############################################################
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






#############################################################
# 2.a Drop columns approach
#############################################################
# Remove object (string) types
drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])
print("MAE from Approach 1 (Drop categorical variables):")
score_dataset(drop_X_train, drop_X_valid, y_train, y_valid) # 17837.82570776256







#############################################################
# 2.b Ordinal Encoder approach
#############################################################
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







#############################################################
# 2.c. One hot encoding approach:
#############################################################

# Investigate Cardinality (Will need to handle cols with high cardinality)
# 'Neighborhood' has cardinality of 25 (25 options)
# can make dataset huge if we one hot encode all of these columns (especially ones with high cardinality)
# only one-hot-encode low cardinality values (ones w/ cardinatlity < 10) - drop others or use ordinal encoding

# Get number of unique entries in each column with categorical data
object_nunique = list(map(lambda col: X_train[col].nunique(), object_cols)) # [5, 2, 4, 4, ...]
pairs = zip(object_cols, object_nunique) # ('Location', 5) ('HasGarage', 2) ...
d = dict(pairs) # turn pairs into nice dictionary 

# Print number of unique entries by column, in ascending order
sorted(d.items(), key=lambda x: x[1]) # sort by the nunique vals [('Street', 2), ('LandSlope', 3), ('LotShape', 4), ...]


# --ONE HOT ENCODING--
# cols that will be one-hot-encoded:
low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10]

# cols that won't be one-hot-encoded (dropped)
high_cardinality_cols = list(set(object_cols) - set(low_cardinality_cols))

# Create encoder
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Performe one-hot-encoding for X_train and X_valid only on low_cardinality cols
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[low_cardinality_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[low_cardinality_cols]))

# one-hot encoding removes index; put it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Remove categorical columns from data (will replace with one-hot encoding)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

# Combine hot encoded cols df with numerical cols df
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

# Ensure all column names have type string 
OH_X_train.columns = OH_X_train.columns.astype(str)
OH_X_valid.columns = OH_X_valid.columns.astype(str)

print("MAE from Approach 3 (One-Hot Encoding):") 
print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid)) # 17525.345719178084
