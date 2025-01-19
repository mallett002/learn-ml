# https://www.kaggle.com/code/williammallettjr/exercise-missing-values/edit
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

# Read the data (These files don't exist, just for documentation)
# Todo: add these files to see how this works
# should use parquet file instead of csv in real world (smaller size. binary format)
X_full = pd.read_csv('../input/train.csv', index_col='Id') # full (train)
X_test_full = pd.read_csv('../input/test.csv', index_col='Id') # test (test)

# Remove rows with missing target, separate target from predictors
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)

# set y (target)
y = X_full.SalePrice

# drop target from training data
X_full.drop(['SalePrice'], axis=1, inplace=True)

# To keep things simple, we'll use only numerical predictors
# set X (features)
X = X_full.select_dtypes(exclude=['object'])
X_test = X_test_full.select_dtypes(exclude=['object'])

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)


############################################################################################
# Step 1 - Preliminary investigation
############################################################################################

# look at fist 5 rows of training data
X_train.head()

# shape of data
print(X_train.shape)

# Number of missing vals in each col of training data
missing_val_count_by_col = X_train.isnull().sum()
print(missing_val_count_by_col[missing_val_count_by_col > 0])




# caclulate MAE
def calc_mae(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)

    model.fit(X_train, y_train)

    predictions = model.predict(X_valid)

    return mean_absolute_error(y_valid, predictions)



############################################################################################
# Step 2 - Drop columns with missing values and see MAE
############################################################################################
# get names of columns with missing values
cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any()]

# drop columns in training and validation data
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)



############################################################################################
# Step 3 - Imputation (replace missing values with averages of that col)
############################################################################################

imputer = SimpleImputer()

imputed_X_train = pd.DataFrame(imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(imputer.transform(X_valid))

# imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

# Normally, imputation does better
# This case, dropping columns did better
# why? Only few values missing in ds, would think imputation would do better..



############################################################################################
# Step 4 - Preprocess train and validation data - Imputation (replace missing values with averages of that col)
############################################################################################
# Preprocessed training and validation features
final_X_train = pd.DataFrame(imputer.fit_transform(X_train))
final_X_valid = pd.DataFrame(imputer.transform(X_valid))

final_X_train.columns = X_train.columns
final_X_valid.columns = X_valid.columns


# Create and train model:
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(final_X_train, y_train)

# Get validation predictions and MAE
preds_valid = model.predict(final_X_train)
print("MAE (Your approach):")
print(mean_absolute_error(y_valid, preds_valid)) # --> 18062.894611872147



############################################################################################
# Step 5 - Use model to predict against completely new dataset (X_test_full dataset)
############################################################################################
# preprocess test data (completely new dataset)
final_X_test = pd.DataFrame(imputer.transform(X_test))
final_X_test.columns = X_test.columns

preds_test = model.predict(final_X_test)

# Save test predictions to file (Can submit this for competition)
output = pd.DataFrame({
    'Id': X_test.index,
    'SalePrice': preds_test
})
output.to_csv('submission.csv', index=False)
