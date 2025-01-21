import pandas as pd
from sklearn.model_selection import train_test_split

# How to deal with categorical vars??
# 3 approaches:
#   1 - Drop them             --> remove them (if not useful)
#   2 - ordinal encoding      --> assigning number
#       - Works well with "ordinal vars" (when there is a clear ranking relationship to vars)
#   3 - on hot endcoding      --> create cols, representing presence/absence of values
#       - works well when no clear ranking to vars (non ordinal) or "nominal vars"


#############################################################
# 1. LOAD SOME DATA
#############################################################

# Read the data
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
low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and # if number of unique is less than 10 and it's an obj (string)
                        X_train_full[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
feature_cols = low_cardinality_cols + numerical_cols
X_train = X_train_full[feature_cols].copy()
X_valid = X_valid_full[feature_cols].copy()

