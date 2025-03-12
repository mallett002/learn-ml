
# How to make a good model
*From Coursera and Kaggle*
1. Training and Validation data
    - Split the data into training data and validation data
    - Each type (training vs validation) has features (X) and targets (y)

2. Prep data
    - Handle missing values
        - drop cols with missing vals or imputation (fill them in with averages)
        - see which one gives lower MAE
    - Feature scaling:
        - aim for -1 <= xj <= 1
            - ex: -100 <= x3 <= 100 (needs scaled)
    - Feature engineering:
        - choose best features for model
        - ex: lot: len & width; We care mostly about lot area len x width (so make that)
    - choose good learning rate
        - start at .001 and increment by x3
    - handle categorical vars
        - Drop them       
        - Ordinal encoding
        - One hot endcoding

3. Train the model
    - Use an ensemble method (random forest or gradient boosting)

4. Make predictions with validation data 

5. Calculate MAE - See how far off from actual targets

6. Find sweet spot with underfitting and overfitting
    - Address underfitting (high bias)
    - Address overfitting (high variance):
        - Get more training data
        - Remove features (just pick the most appropriate Xs)
        - *Regularization

7. Using pipelines can greatly clean up code. See kaggle/04_intermediate_ml/03_pipelines.py

8. Be aware of data leakage
    - Is the model cheating in any way with the validation data?
        - Target Leakage: Training data has info that won't be available when actually predicting in real world
            "Avoid using features in training that are directly or indirectly correlated to the target **after the event has occurred**"
        - Train-Test Contamination: Allowing test (validation) data a part in the actual training process
            "Always separate train/test sets and ensure preprocessing steps don’t use information from the test set during training"

Try out a competition on kaggle: https://www.kaggle.com/competitions
