import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

X = pd.read_csv('some/path/to/csv.csv', index_col='Id')
X_test = pd.read_csv('some/other/path/to/csv.csv', index_col='Id')

# ##################################
# Dealing with Categorical Data #
# ##################################

# **Need to remove or encode categorical data before training model**

# Remove rows with missing target
X.dropna(axis=0, subset=['SalePrice'], inplace=True)

# Separate target from predictors
y = X.SalePrice
X.drop(['SalePrice'], axis=1, inplace=True) #  remove the sale price col from the rows (axis 1 = col)

# keep things simple - drop cols with missing vals
cols_with_missing = [
    col for col in X.columns
    if bool(X[col].isnull().any())
]

X.drop(cols_with_missing, axis=1, inplace=True)

# separate training & validation data
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=0)


def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)


# ############################################
# Approach #1. Drop cols with categorical data
# ############################################

# Drop columns in training and validation data:
drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])

# See MAE from droping categorical vars:
print("MAE from Approach 1 (Drop categorical variables):")
print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))

# ############################################
# Approach #2. Ordinal encoding
# ############################################

# **If you have values in validation data that don't occur in training, ordinal encoding will throw an error**
# Can just drop these columns as well so it's not a problem (but you might be losing some good data)

# Get categorical columns:
object_cols = [
    col
    for col in X_train.columns
    if X_train[col].dtype == "object"
]

# Columns that can be safely ordinal encoded:
# set() -> get the unique values
# A.issubset(B) -> all values in A also occur in B 
good_label_cols = [
    col
    for col in object_cols
    if set(X_valid[col]).issubset(set(X_train[col]))
]

# Problematic columns that will be dropped from the dataset:
bad_label_cols = list(set(object_cols) - set(good_label_cols))
# {A, B, C} - {A} -> {B, C}

# Drop categorical values that won't be encoded
# Create new DFs with bad cols removed
label_X_train = X_train.drop(bad_label_cols, axis=1)
label_X_valid = X_valid.drop(bad_label_cols, axis=1)

# Apply ordinal encoder:
encoder = OrdinalEncoder()

label_X_train[good_label_cols] = encoder.fit_transform(X_train[good_label_cols])
label_X_valid[good_label_cols] = encoder.fit(X_valid[good_label_cols])

# See MAE from ordinal encoding:
print("MAE from Approach 2 (Ordinal Encoding):") 
print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))

# ############################################
# Investigating Cardinality
# ############################################
 # create list of the number of uniques ex [25, 8, 6]
object_nunique = list(map(lambda col: X_train[col].nunique(), object_cols))

# zip together object cols and the object_nunique (by position) and then put in dictionary for easy lookup:
uniques_by_cols = dict(zip(object_cols, object_nunique))
# ex:
# {
#   'Neighborhood': 25,
#   'Style': 8,
#   'RoofType': 6
# }

# sort them ascending:
sorted(uniques_by_cols.items(), key=lambda x: x[1])

# High cardinality (ones with cardinality > ~10) should be dropped, or just use ordinal encoding. One hot encoding can greatly increase data size

# Find cols that have low/high cardinality
low_cardinality_cols = [
    col
    for col in object_cols
    if X_train[col].nunique() < 10
]

high_cardinality_cols = list(set(object_cols) - set(low_cardinality_cols))

# ############################################
# Approach 3. One Hot Encoding
# ############################################

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[low_cardinality_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[low_cardinality_cols]))

# Side note - AI said I can do this so we don't need to add back indexes, nor need to ensure all column names are strings down below as well:
# OH_cols_train = pd.DataFrame(
#     OH_encoder.fit_transform(X_train[low_cardinality_cols]),
#     columns=OH_encoder.get_feature_names_out(low_cardinality_cols),
#     index=X_train.index
# )

# One hot encoding removed the index; put it back
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

# See MAE from ordinal encoding:
print("MAE from Approach 3 (One-Hot Encoding):") 
print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))
