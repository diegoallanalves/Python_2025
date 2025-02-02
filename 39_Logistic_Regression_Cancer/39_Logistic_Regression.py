#import pandas

# This code imports the pandas library and assigns it the alias "pd".
# It then creates a list of column names for a dataset called "pima".
# The dataset is loaded from a CSV file called "pima-indians-diabetes.csv" using the pandas function "read_csv".
# The "header=None" argument specifies that the CSV file does not have a header row, and the "names=col_names" argument assigns the column names from the previously defined list to the dataset.

import pandas as pd
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
# load dataset
pima = pd.read_csv("C:\\Users\\diego\\Desktop\\python\\39_Logistic_Regression_Cancer\\diabetes.csv", header=None, names=col_names)

# This code is written in Python.
# The pima.head() function is used to display the first few rows of a Pandas DataFrame called pima.
# This is a common way to quickly inspect the data and get a sense of what it looks like.
# By default, head() displays the first 5 rows of the DataFrame, but you can specify a different number of rows by passing an argument to the function (e.g.
# pima.head(10) would display the first 10 rows).
print(pima.head())

#pima["pregnant"] = [float(str(i).replace(" ", "")) for i in pima["pregnant"]]

# This code is written in Python.
# The code is splitting a dataset into two parts: features and target variable.
# The feature_cols variable is a list of column names that represent the features in the dataset.
# The X variable is assigned the values of the columns specified in feature_cols, which represents the features.
# The y variable is assigned the values of the label column in the pima dataset, which represents the target variable.
# In summary, this code is selecting specific columns from a dataset to use as features and assigning the values of the target variable to a separate variable.
# split dataset in features and target variable
feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
X = pima[feature_cols] # Features
y = pima.label # Target variable

# Splitting Data
# To understand model performance, dividing the dataset into a training set and a test set is a good strategy.
# Let's split the dataset by using the function train_test_split(). You need to pass 3 parameters: features, target, and test_set size. Additionally, you can use random_state to select records randomly.

# split X and y into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)

# This code is splitting the data into training and testing sets using the train_test_split function from the sklearn.model_selection module.
# The X and y variables are the input features and target variable, respectively.
# The test_size parameter specifies the proportion of the data that should be used for testing, in this case 25%.
# The random_state parameter sets the seed for the random number generator, ensuring that the same split is obtained each time the code is run.
# The function returns four arrays: X_train and y_train are the training sets, while X_test and y_test are the testing sets.
# These arrays can then be used to train and evaluate a machine learning model.

# Result:
# Here, the Dataset is broken into two parts in a ratio of 75:25. It means 75% data will be used for model training and 25% for model testing.

# Model Development and Prediction
# First, import the Logistic Regression module and create a Logistic Regression classifier object using the LogisticRegression() function with random_state for reproducibility.
# Then, fit your model on the train set using fit() and perform prediction on the test set using predict().
# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression(random_state=16, solver='lbfgs', max_iter=3000)

# fit the model with data
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

# This code is using the scikit-learn library in Python to perform logistic regression.
# First, the code imports the LogisticRegression class from the sklearn.linear_model module.
# Next, an instance of the LogisticRegression class is created with the random_state parameter set to 16.
# This sets the seed for the random number generator used in the model, ensuring that the results are reproducible.
# Then, the fit() method is called on the logreg object with the training data X_train and y_train as arguments.
# This trains the logistic regression model on the training data.
# Finally, the predict() method is called on the logreg object with the test data X_test as an argument.
# This generates predictions for the test data based on the trained model, which are stored in the y_pred variable.

# Model Evaluation using Confusion Matrix
# A confusion matrix is a table that is used to evaluate the performance of a classification model.
# You can also visualize the performance of an algorithm. The fundamental of a confusion matrix is the number of correct and incorrect predictions summed up class-wise.
# import the metrics class
from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)

# This code imports the metrics class from the sklearn library, which is used for evaluating the performance of machine learning models.
# Then, it calculates the confusion matrix by calling the confusion_matrix() function from the metrics class, passing in the true labels y_test and the predicted labels y_pred as arguments.
# The confusion matrix is a table that summarizes the performance of a classification algorithm by comparing the predicted and actual class labels for a set of test data.
# Finally, the code prints the confusion matrix to the console.
# This code creates a 2D NumPy array with dimensions 2x2.
# The values in the array are 115, 8 in the first row and 30, 39 in the second row.
# The array can be accessed and manipulated using NumPy array methods and functions.
# Here, you can see the confusion matrix in the form of the array object.
# The dimension of this matrix is 2*2 because this model is binary classification.
# You have two classes 0 and 1. Diagonal values represent accurate predictions, while non-diagonal elements are inaccurate predictions. In the output, 115 and 39 are actual predictions, and 30 and 8 are incorrect predictions.

# Visualizing Confusion Matrix using Heatmap
# Let's visualize the results of the model in the form of a confusion matrix using matplotlib and seaborn.
# Here, you will visualize the confusion matrix using Heatmap.

# import required modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

# Text(0.5,257.44,'Predicted label')
# This code imports the required modules numpy, matplotlib.pyplot, and seaborn.
# It then defines a list of class names and creates a subplots object using matplotlib.
# It also creates tick marks for the x and y axes using numpy's arange function and sets the tick labels to the class names.
# The code then creates a heatmap using seaborn's heatmap function, passing in a pandas DataFrame containing the confusion matrix as an argument.
# The heatmap is annotated with the values in the matrix and colored using the YlGnBu colormap.
# The x-axis label is set to be on top of the plot using the xaxis.set_label_position method, and the plot is tightened using the tight_layout method.
# The title of the plot is set to "Confusion matrix" with a y-coordinate of 1.1, and the y-axis and x-axis labels are set to "Actual label" and "Predicted label", respectively.

# Confusion Matrix Evaluation Metrics
# Let's evaluate the model using classification_report for accuracy, precision, and recall.
from sklearn.metrics import classification_report
target_names = ['without diabetes', 'with diabetes']
print(classification_report(y_test, y_pred, target_names=target_names))

# This code imports the classification_report function from the sklearn.metrics module.
# It then defines a list of target names for the classification report.
# Finally, it prints the classification report by passing in the y_test and y_pred variables as arguments, along with the target_names list.
# The classification_report function generates a report that includes precision, recall, F1-score, and support for each class in the target_names list.
# The y_test and y_pred variables are assumed to be arrays of true and predicted labels, respectively, for a binary classification problem.
# The target_names list is used to label the two classes in the report.
# This code snippet shows the performance metrics of a binary classification model.
# The model has predicted whether a person has diabetes or not, and the metrics are calculated based on the comparison of the predicted values with the actual values.
# The precision, recall, and f1-score are three commonly used metrics to evaluate the performance of a binary classification model.
# Precision measures the proportion of true positives among all the positive predictions, while recall measures the proportion of true positives among all the actual positives.
# F1-score is the harmonic mean of precision and recall.
# The support column shows the number of samples in each class.
# In this case, there are 123 samples without diabetes and 69 samples with diabetes.
# The accuracy is the proportion of correct predictions among all the predictions.
# The macro avg and weighted avg are the average metrics across all the classes, with the former giving equal weight to each class and the latter giving more weight to the class with more samples.
# Overall, this code snippet provides a summary of the performance of a binary classification model, which can be used to evaluate and compare different models.

# Well, you got a classification rate of 80%, considered as good accuracy.

# Precision: Precision is about being precise, i.e., how accurate your model is. In other words, you can say, when a model makes a prediction, how often it is correct. In your prediction case, when your Logistic Regression model predicted patients are going to suffer from diabetes, that patients have 73% of the time.

# Recall: If there are patients who have diabetes in the test set and your Logistic Regression model can identify it 57% of the time.

# ROC Curve
# Receiver Operating Characteristic(ROC) curve is a plot of the true positive rate against the false positive rate. It shows the tradeoff between sensitivity and specificity.

y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

# This code is used to plot a Receiver Operating Characteristic (ROC) curve for a logistic regression model.
# First, the predict_proba method of the logistic regression model (logreg) is used to predict the probabilities of the positive class for the test set (X_test).
# The probabilities for the positive class are extracted using [::,1].
# Next, the roc_curve function from the metrics module is used to calculate the false positive rate (fpr) and true positive rate (tpr) for different probability thresholds.
# The y_test parameter is the true labels for the test set.
# Then, the roc_auc_score function from the metrics module is used to calculate the area under the ROC curve (auc) for the predicted probabilities and true labels.
# Finally, the plot function from the pyplot module is used to plot the ROC curve using fpr and tpr.
# The label parameter is used to add a label to the plot with the calculated auc value.
# The legend function is used to display the label in the plot.
# The show function is used to display the plot.

# AUC score for the case is 0.88. AUC score 1 represents a perfect classifier, and 0.5 represents a worthless classifier.

# Conclusion
# In this tutorial, you covered a lot of details about Logistic Regression. You have learned what logistic regression is, how to build respective models, how to visualize results and some of the theoretical background information. Also, you covered some basic concepts such as the sigmoid function, maximum likelihood, confusion matrix, ROC curve.

# Hopefully, you can now utilize the Logistic Regression technique to analyze your own datasets. Thanks for reading this tutorial!