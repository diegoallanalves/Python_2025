# Fail

import re

import numpy as np

import tweepy

from tweepy import OAuthHandler

from textblob import TextBlob

import matplotlib.pyplot as plt

import pandas as pd

from wordcloud import WordCloud

from better_profanity import profanity

# keys and tokens from the Twitter Dev Console
consumer_key = 'Jqcyy66fQtA6TDqn8fNKbwn8C'
consumer_secret = 'QuAUL9WnQWEdp74OkfgpvQbCTJAKdImQ576OTrEaOOfikjwZLx'
access_token = '1467797186540888070-mewg98PE3ic1uEGFgAEqosafuH5ZA7'
access_token_secret = '3bWSVtVt5nWkvq73ZtNzxLlFdLmvhZ5TvpeIhN5eIky5G'

# Access Twitter Data

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

# Input a query from the user

query = input("SMMT: ")

# In this case, we will input the query as 'Elon Musk'

# Filter the query to remove retweets

filtered = query + "-filter:retweets"

# Generate the latest tweets on the given query

tweets = tweepy.Cursor(api.search_tweets,
                           q=filtered,
                           lang="en").items(100)

# Create a list of the tweets, the users, and their location

list1 = [[tweet.text, tweet.user.screen_name, tweet.user.location] for tweet in tweets]

# Convert the list into a dataframe

df = pd.DataFrame(data=list1,
                    columns=['tweets','user', "location"])
# Convert only the tweets into a list

tweet_list = df.tweets.to_list()

# Create a function to clean the tweets. Remove profanity, unnecessary characters, spaces, and stopwords.

def clean_tweet(tweet):
    if type(tweet) == np.float:
        return ""
    r = tweet.lower()
    r = profanity.censor(r)
    r = re.sub("'", "", r) # This is to avoid removing contractions in english
    r = re.sub("@[A-Za-z0-9_]+","", r)
    r = re.sub("#[A-Za-z0-9_]+","", r)
    r = re.sub(r'http\S+', '', r)
    r = re.sub('[()!?]', ' ', r)
    r = re.sub('\[.*?\]',' ', r)
    r = re.sub("[^a-z0-9]"," ", r)
    r = r.split()
    stopwords = ["for", "on", "an", "a", "of", "and", "in", "the", "to", "from"]
    r = [w for w in r if not w in stopwords]
    r = " ".join(word for word in r)
    return r

cleaned = [clean_tweet(tw) for tw in tweet_list]
print(cleaned)


