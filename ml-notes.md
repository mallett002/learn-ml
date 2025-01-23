
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
            - ex: -100 <= x3 <= 100 (should scale)
    - choose good learning rate
        - start at .001 and increment by x3
    - handle categorical vars
        - Drop them       
        - Ordinal encoding
        - One hot endcoding 3 approaches:

3. Train the model

4. Make predictions with validation data 

5. Calculate MAE - See how far off from actual targets

6. Find sweet spot with underfitting and overfitting


