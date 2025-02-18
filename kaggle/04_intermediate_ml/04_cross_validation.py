# https://www.kaggle.com/code/alexisbcook/cross-validation
#1. Divide the Dataset: Split your dataset into 5 equal parts (folds), each containing roughly 20% of the data.
#2. Validation and Training: For each fold, you use it as the validation set, and the remaining 4 folds (80% of the data) are used for training.
#3. Iterate: Repeat this process 5 times, each time with a different fold as the validation set.
#4. Evaluate: After all 5 iterations, you'll have 5 different validation scores. You can then average these scores to get an overall performance metric for your model.
# This method helps ensure that your model performs well on different subsets of the data, making it more robust and less likely to overfit.
# Better for training smaller datasets

import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score


# Read the data
data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')

# Select subset of predictors
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]

# Select target
y = data.Price



# Define a pipeline that uses an imputer to fill in missing values and a random forest model to make predictions
my_pipeline = Pipeline(steps=[
    ( 'preprocessor', SimpleImputer() ),
    ( 'model', RandomForestRegressor(n_estimators=50, random_state=0))
])



# Cross validate the data
# Multiply by -1 since sklearn calculates *negative* MAE
scores = -1 * cross_val_score(my_pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')
# Can choose other scoring mechanisms: https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules

print("MAE scores:\n", scores)

print("Average MAE score (across experiments):")
print(scores.mean())

# Note that we no longer need to keep track of separate training and validation sets

