# How to implement smote technique: https://medium.com/@corymaklin/synthetic-minority-over-sampling-technique-smote-7d419696b88c

'''SMOTE: Synthetic Minority Oversampling Technique
SMOTE is an oversampling technique where the synthetic samples are generated for the
minority class. This algorithm helps to overcome the overfitting problem posed by random oversampling.'''

from random import randrange, uniform
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score

# In this example, we will make use of the Credit Card Fraud Detection
# dataset on Kaggle to train a model to determine
# whether a given transaction is fraudulent. We read the CSV file
# and store its contents in a Pandas DataFrame as follows:
df = pd.read_csv("C:\\Users\\diego\\Desktop\\python\\37_Smote_Technique\\coffee_analysis.csv")

# Print the data:
print(df.head(5))
df = df.dropna()

# As we can see, there are significantly more negative samples than positive samples.
print(df['rating'].value_counts())

# For simplicity, we remove the time dimension.
df = df.drop(['origin'], axis=1)

# We split the dataset into features and labels.
X = df.drop(['rating'], axis=1)
y = df['rating']

# In order to evaluate the performance of our model, we split the data into training and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Next, we initialize an instance of the RandomForestClassifier class.
rf = RandomForestClassifier(random_state=42)

# We fit our model to the training set.
rf.fit(X_train, y_train)

# Finally, we use our model to predict whether a transaction is fraudulent given what it has learnt.
y_pred = rf.predict(X_test)

# Suppose our dataset contained 100 examples of fraudulent and 9900 examples of regular transactions.
# If we used accuracy to measure the model’s performance, it could obtain an accuracy of 99% by simply predicting false every time.
# It’s for this reason that we use a confusion matrix to evaluate the model’s performance. As we can see,
# our model classified 23 samples as non-fraudulent when, in fact, they were.
print(confusion_matrix(y_test, y_pred))

# For ease of comparison, if we wanted a single number to gauge the model’s performance, we could use recall.
# Recall (dad joke) that recall is equal to the number of true positives divided by the sum of true positives and false negatives.
print(recall_score(y_test, y_pred))

print("Use Smothe with Python Library ##################################################################################################################################################")
# SMOTE using library: We can then import the SMOTE class.
from imblearn.over_sampling import SMOTE

# To avoid confusion, we read the csv file again.
df = pd.read_csv("C:\\Users\\diego\\Desktop\\python\\37_Smote_Technique\\coffee_analysis.csv")
df = df.dropna()
# For simplicity, we remove the time dimension.
df = df.drop(['origin'], axis=1)
# We split the dataset into features and labels.
print(df['rating'].value_counts())
X = df.drop(['rating'], axis=1)
y = df['rating']

# We instantiate an instance of the SMOTE class.
# It’s worth noting that, by default, it will ensure that there are an equal number of positive samples as negative samples.
sm = SMOTE(random_state=42, k_neighbors=5)

# We apply the SMOTE algorithm to the dataset as follows:
X_res, y_res = sm.fit_resample(X, y)

print("Smote data replication for X")
print(X_res)
print("Smote data replication for Y")
print(y_res)
# Again, we split the dataset, train the model and predict whether the samples in the testing dataset should be considered fraudulent or not.
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# If we look at the confusion matrix, we can see that there are an equal number of positive samples as negative samples
# and the model didn’t have any false negatives. Ergo, the recall is 1.
print(confusion_matrix(y_test, y_pred))
print(recall_score(y_test, y_pred))

# Conclusion
# When a machine learning model is trained on an imbalanced dataset it tends to perform poorly.
# When acquiring more data isn’t an option, we have to resort to down-sampling or up-sampling.
# Down-sampling is bad because it removes samples that could otherwise have been used to train the model.
# Up-sampling on its own is less than ideal since it causes our model to overfit.
# SMOTE is a technique to up-sample the minority classes while avoiding overfitting.
# It does this by generating new synthetic examples close to the other points (belonging to the minority class) in feature space.





