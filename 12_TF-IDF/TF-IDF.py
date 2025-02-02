#Importing Libraries
import numpy as np 
import pandas as pd 
import re  
import nltk 
nltk.download('stopwords')  
from nltk.corpus import stopwords 

# Loading Dataset
tweets = pd.read_csv("https://raw.githubusercontent.com/kolaveridi/kaggle-Twitter-US-Airline-Sentiment-/master/Tweets.csv")

print(tweets.head())

# Similarly, to find the number of rows and columns in the dataset, you can use the shape attribute as shown below:
print(tweets.shape)

# Exploratory Data Analysis
import seaborn as sns
from matplotlib import pyplot as plt

# Plot the data into three types of plots, however the last one is the most important one.
# Plot 1
#sns.countplot(x='airline_sentiment', data=tweets)

# Plot 2
#sns.countplot(x='airline', data=tweets)

# Plot 3
sns.countplot(x='airline', hue="airline_sentiment", data=tweets)
plt.show()

# Data Pre-processing

# Let’s divide our dataset into features and label set
X = tweets.iloc[:, 10].values
y = tweets.iloc[:, 1].values

# Our dataset contains many special characters and empty spaces. You need to remove them in order to have a clean dataset
processed_tweets = []

for tweet in range(0, len(X)):
    # Remove all the special characters
    processed_tweet = re.sub(r'\W', ' ', str(X[tweet]))

    # remove all single characters
    processed_tweet = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_tweet)

    # Remove single characters from the start
    processed_tweet = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_tweet)

    # Substituting multiple spaces with single space
    processed_tweet = re.sub(r'\s+', ' ', processed_tweet, flags=re.I)

    # Removing prefixed 'b'
    processed_tweet = re.sub(r'^b\s+', '', processed_tweet)

    # Converting to Lowercase
    processed_tweet = processed_tweet.lower()

    processed_tweets.append(processed_tweet)

# Dividing Data to Training and Test Sets
from sklearn.feature_extraction.text import TfidfVectorizer
tfidfconverter = TfidfVectorizer(max_features=2000, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
X = tfidfconverter.fit_transform(processed_tweets).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training and Evaluating the Text Classification Model
from sklearn.ensemble import RandomForestClassifier
text_classifier = RandomForestClassifier(n_estimators=100, random_state=0)
text_classifier.fit(X_train, y_train)

predictions = text_classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
print(accuracy_score(y_test, predictions))