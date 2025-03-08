# Data leakage: Is your model cheating on the training?

# Data leakage happens when your training data includes information about the target,
# but similar data will not be available when the model is used for prediction.
# This leads to high performance on the training set (and possibly even the validation data), but the model will perform poorly in production.
# This causes the model to perform well on the validation data, but poorly in production.

# 1. Target Leakage: "Predictors include data that will not be available at the time you make predictions"
# Ex: Predicting who will get pnemonia
#   - feature in training set (`took_antibiotics`)
#       - Typically take this after actually getting pnemonia
#       - Model learns this strong correlation, `took_antibiotics` --> has_pnemonia = true
#   - `took_antibiotics` will be false in real world bc trying to predict who will get pnemonia
#   - Should remove it from training set



# 2. Train-Test Contamination
# ex: Imputation before calling train_test_split() (for splitting training and validation data)
# Fix: Don't let validation data have any part in the training process, other than just to see how well the model does with new data
# Real-world parallel: 
#   - Imagine you're trying to predict tomorrow's weather.
#   - If you're training a model, you should only use historical weather data (like averages up to today), not tomorrow's data, to tune your predictions

# Fix in practice:
#   - Training the model: first split data into training and validation data.
#   - Impute on the training data (to generate the mean).
#   - Then, when you are testing the model with the validation data, use that same mean you got from the training data to fill in missing values

import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Read the data
data = pd.read_csv('../input/aer-credit-card-data/AER_credit_card_data.csv', 
                   true_values = ['yes'], false_values = ['no'])

# Select target
y = data.card

# Select predictors
X = data.drop(['card'], axis=1)

# Small dataset, so use cross-validation to ensure accurate measures of model quality
my_pipeline = make_pipeline(
    RandomForestClassifier(n_estimators=100)
)

cv_scores = cross_val_score(my_pipeline, X, y, cv=5, scoring='accuracy')

print("Cross-validation accuracy: %f" % cv_scores.mean()) # Cross-validation accuracy: 0.981052

# 98% is high, very sus. Need to inspect for target leakage.
