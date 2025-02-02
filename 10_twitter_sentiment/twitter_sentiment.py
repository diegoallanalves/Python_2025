# Import the libraries
import tweepy
from textblob import TextBlob
from wordcloud import WordCloud
import re
import matplotlib.pyplot as plt
import sys
plt.style.use('fivethirtyeight')

from sre_parse import CATEGORIES
import pandas as pd

# Twitter credentials for the app
# initialize api instance
consumer_key='J7GxhVVerU6TFye4hc23EVZE7'
consumer_secret='SgOgfvAQlxTaWpl1JmZUc3tfYUClFwABSR7tP2QvvhBEwZ7url'
access_token_key='1467797186540888070-ThejW6VLGDch5m0CWkIFW7Iy8WGXf2'
access_token_secret='bsQ7HybkXE94OjICKnq7l6LUhI0nIoOxCuf4Zo7rUqnTa'

# Creating the authentication object
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
# Setting your access token and secret
auth.set_access_token(access_token_key, access_token_secret)
# Creating the API object while passing in auth information
api = tweepy.API(auth)

# Using the API object to get tweets from your timeline, and storing it in a variable called public_tweets
public_tweets = api.home_timeline()
# foreach through all tweets pulled
for tweet in public_tweets:
# printing the text stored inside the tweet object
   print(tweet.text)

# Creating the API object while passing in auth information
api = tweepy.API(auth)

# The Twitter user who we want to get tweets from
name = "Trump"
# Number of tweets to pull
tweetCount = 100

# Calling the user_timeline function with our parameters
results = api.user_timeline(id=name, count=tweetCount)

# foreach through all tweets pulled
for tweet in results:
   # printing the text stored inside the tweet object
   print(tweet.text)

# Creating the API object while passing in auth information
api = tweepy.API(auth)

# The search term you want to find
query = "Trump"
# Language code (follows ISO 639-1 standards)
language = "pt"

# Calling the user_timeline function with our parameters
results = api.search(q=query, lang=language)

# foreach through all tweets pulled
for tweet in results:
   # printing the text stored inside the tweet object
   print(tweet.user.screen_name,"Tweeted:",tweet.text)

#  Print the last 5 tweets
print("Show the 5 recent tweets:\n")
i=1
for tweet in results[:100]:
    print(str(i) +') '+ tweet.text + '\n')
    i= i+1

# Create a dataframe with a column called Tweets
bolsonaro_100_tweets = pd.DataFrame([tweet.text for tweet in results], columns=['Tweets'])
# Show the first 100 rows of data
#bolsonaro_100_tweets.head()

print(bolsonaro_100_tweets)


# Create a function to clean the tweets
def cleanTxt(text):
    text = re.sub('@[A-Za-z0–9]+', '', text)  # Removing @mentions
    text = re.sub('#', '', text)  # Removing '#' hash tag
    text = re.sub('RT[\s]+', '', text)  # Removing RT
    text = re.sub('https?:\/\/\S+', '', text)  # Removing hyperlink

    return text

# Clean the tweets
bolsonaro_100_tweets['Tweets'] = bolsonaro_100_tweets['Tweets'].apply(cleanTxt)

# Show the cleaned tweets
print(bolsonaro_100_tweets)

# Create a function to get the subjectivity
def getSubjectivity(text):
   return TextBlob(text).sentiment.subjectivity

# Create a function to get the polarity
def getPolarity(text):
   return  TextBlob(text).sentiment.polarity


# Create two new columns 'Subjectivity' & 'Polarity'
bolsonaro_100_tweets['Subjectivity'] = bolsonaro_100_tweets['Tweets'].apply(getSubjectivity)
bolsonaro_100_tweets['Polarity'] = bolsonaro_100_tweets['Tweets'].apply(getPolarity)

# Show the new dataframe with columns 'Subjectivity' & 'Polarity'
print(bolsonaro_100_tweets)

# word cloud visualization
allWords = ' '.join([twts for twts in bolsonaro_100_tweets['Tweets']])
wordCloud = WordCloud(width=500, height=300, random_state=21, max_font_size=110).generate(allWords)

plt.imshow(wordCloud, interpolation="bilinear")
plt.axis('off')
plt.show()

# Create a function to compute negative (-1), neutral (0) and positive (+1) analysis
def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'
bolsonaro_100_tweets['Analysis'] = bolsonaro_100_tweets['Polarity'].apply(getAnalysis)

# Show the dataframe
print(bolsonaro_100_tweets)

# Printing positive tweets
print('Printing positive tweets:\n')
j=1
sortedDF = bolsonaro_100_tweets.sort_values(by=['Polarity']) #Sort the tweets
for i in range(0, sortedDF.shape[0] ):
  if( sortedDF['Analysis'][i] == 'Positive'):
    print(str(j) + ') '+ sortedDF['Tweets'][i])
    print()
    j = j+1

# Printing negative tweets
print('Printing negative tweets:\n')
j=1
sortedDF = bolsonaro_100_tweets.sort_values(by=['Polarity'],ascending=False) #Sort the tweets
for i in range(0, sortedDF.shape[0] ):
  if( sortedDF['Analysis'][i] == 'Negative'):
    print(str(j) + ') '+sortedDF['Tweets'][i])
    print()
    j = j+1

# Plotting
plt.figure(figsize=(4,4))
for i in range(0, bolsonaro_100_tweets.shape[0]):
  plt.scatter(bolsonaro_100_tweets["Polarity"][i], bolsonaro_100_tweets["Subjectivity"][i], color='Blue')
# plt.scatter(x,y,color)
plt.title('Sentiment Analysis')
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')
plt.show()

# Print the percentage of positive tweets
ptweets = bolsonaro_100_tweets[bolsonaro_100_tweets.Analysis == 'Positive']
ptweets = ptweets['Tweets']
print(ptweets)

positive_tweets_percentage = round( (ptweets.shape[0] / bolsonaro_100_tweets.shape[0]) * 100 , 1)

print(positive_tweets_percentage)

# Print the percentage of negative tweets
ntweets = bolsonaro_100_tweets[bolsonaro_100_tweets.Analysis == 'Negative']
ntweets = ntweets['Tweets']
print(ntweets)

negative_tweets_percentage = round((ntweets.shape[0] / bolsonaro_100_tweets.shape[0]) * 100, 1)

print(negative_tweets_percentage)

# Show the value counts
print(bolsonaro_100_tweets['Analysis'].value_counts())

# Plotting and visualizing the counts
plt.title('Sentiment Analysis')
plt.xlabel('Sentiment')
plt.ylabel('Counts')
bolsonaro_100_tweets['Analysis'].value_counts().plot(kind = 'bar')
plt.show()

sys.exit()



