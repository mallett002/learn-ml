Left off: https://www.coursera.org/learn/machine-learning/lecture/1Z0TT/cost-function-formula


# Supervised Learning
Give: x (input) --> y (output label)
learns from being given right answers (y)
x is input, y is the answer.

### Regression
- predict number based on inputs
- many different numbers to choose from
- ex: pick house price based on square footage

### Classification 
different: trying to predict small # of categories
- ex: 0, or 1 (true or false)
- Put them into categories
classes & categories are same thing
- finite amount of categrories (outcomes)




# Unsupervised Learning
Not given labels (Y)
algorithm finds something interesting in unlabled data
ask algorithm, what can you find that is intersting

### Clustering
puts data into several clusters
here's a bunch of data, find structure in the data
ex: grouping customers

### Anomaly detection
unusual events
-ex: fraud

### Dimensionality reduction
compress data using fewer numbers




# Supervised learning - Linear regression
predicts numbers - regression

### Terminology
Training Set - data used to train the model
- x or feature

Target variable - the data you are trying to teach the model to reach
- y or output

m = total number of training examples
(x, y) = single training example



# Gradient Descent with Logistic Regression
how to find good params for w,b
once find, can use model to find y-hat