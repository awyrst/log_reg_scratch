#Binary Classification on Chronic Kidney Disease and Defective Battery Datasets

Refer to my website for a quick description of the objectives and results - https://davwyr.com/portfolio/Projects/ML_projects/log_regression_proj/log_regression_proj.html

We only experimented with a learning rate decay that consisted of dividing the initial learning rate by k+1, where k is the iteration of gradient descent.

When it came to removing features, we would fit the model, take the absolute value of the parameters/weights, take the mean of those absolute values, and then
divide every absolute weight value by that mean. If the resulting value was less then a certain threshold percentage of that mean, then we would remove them.
This is how we were evaluating whether a feature was useful or not. The two thresholds we used were 30% and 120%.

models.py: 

All the code with special functions and classes is in this file. The "model" class has a field for every key 
component of a logistic regression model. The features X, the labels y, the weights w, a learning rate, another parameter
that specifies how the learning rate changes over time, and a parameter that specifies the maximum number of steps of
gradient descent to run. It also has built-in methods to normalize the data, add fictitious quadratic and cubic terms,
and a method to remove features as well.

There is a function called "fit", which takes in an instance of a model and runs gradient descent on the loss function
to arrive at the optimized values for the weights W.

There is a function called predict, which takes in a model, and outputs the predicted labels.

There is a function Accu_eval, that takes in the actual labels and predicted labels, and produces an accuracy score.

The crossvalidate10 function takes in an array of T models, runs 10-fold cross-validation on them all, and outputs
an array of each of their cross-validation accuracies.

We also have an additional class called "preprocess_visualize", which takes in the data from the csv files,
and has built-in functions to randomize the data and display statistics on each feature's distribution.

Every model that was run had a maximum number of steps of 500 in gradient descent, and the gradient descent would
terminate if the percent difference in the norm of the weight vector dropped below 0.1%. We elaborate more on this
in the report.


crossvalidate_accuracies.py: 

This model generates every model we use in the comparison, runs 10-fold cross-validation on them all, and outputs
a csv file comparing their accuracies on both datasets.
