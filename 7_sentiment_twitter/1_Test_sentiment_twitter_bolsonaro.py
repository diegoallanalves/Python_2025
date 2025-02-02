# Basic Imports
import tweepy
import pandas as pd
import numpy as np
import time
import os
import re

# Plotting and Visualization
import matplotlib.pyplot as plt
from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS

from IPython import get_ipython

ipy = get_ipython()
if ipy is not None:
    ipy.run_line_magic('matplotlib', 'inline')

# %matplotlib inline

# import nltk
# nltk.download()

# TextBlob Imports
from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier

# NLTK Imports
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# MAP
from geopy.geocoders import Nominatim
import folium
from folium import plugins
from geopy.geocoders import Nominatim

tweets_df = pd.read_csv("lula-bolsonaro.csv")

print(tweets_df.head())

# Top Searched Tweets
fav_max = np.max(tweets_df['Likes'])
rt_max = np.max(tweets_df['RTs'])

fav = tweets_df[tweets_df.Likes == fav_max].index[0]
rt = tweets_df[tweets_df.RTs == rt_max].index[0]

# Max FAVs:
print("O tweet com mais curtidas é: \n{}".format(tweets_df['Tweets'][fav]))
print("Número de curtidas: {}".format(fav_max))
print("\n")
# Max RTs:
print("O tweet com mais retweet é: \n{}".format(tweets_df['Tweets'][rt]))
print("Número de retweets: {}".format(rt_max))

# Source of the Tweets
sources = []
for source in tweets_df['Source']:
    if source not in sources:
        sources.append(source)

percent = np.zeros(len(sources))

for source in tweets_df['Source']:
    for index in range(len(sources)):
        if source == sources[index]:
            percent[index] += 1
            pass

newDF = pd.DataFrame({
    'source': percent,
}, index=sources)

sources_sorted = newDF.sort_values('source', ascending=False)
ax = sources_sorted.source.plot(figsize=(13,6), kind='barh', color='#002060')
ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

print(plt.show())

# Sentiment Analysis in Português and English

base_path = 'C:\\Users\\diego\\Desktop\\data_set\\ReLi-Lex'
train = []
wordsPT = []
wordsPT_sentiments = []

files = [os.path.join(base_path, f) for f in os.listdir(base_path)]

for file in files:
    t = 1 if '_Positivos' in file else -1
    with open(file, 'r') as content_file:
        content = content_file.read()
        all = re.findall('\[.*?\]', content)
        for w in all:
            wordsPT.append((w[1:-1]))
            wordsPT_sentiments.append(t)
            train.append((w[1:-1], t))

cl = NaiveBayesClassifier(train)


def clean_tweet(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())


def sentiment(tweet):
    polarity = 0

    # blob = TextBlob(tweet, classifier=cl)
    blob = TextBlob(clean_tweet(tweet), classifier=cl)  # Clean Tweets

    for s in blob.sentences:
        polarity = s.classify() + polarity

    if polarity > 0:
        return 1
    elif polarity < 0:
        return -1
    else:
        return 0


def analize_sentimentEN(tweet):
    analysis = TextBlob(tweet)

    if analysis.detect_language() != 'en':
        analysis = TextBlob(str(analysis.translate(to='en')))
        time.sleep(0.5)

    if analysis.sentiment.polarity > 0:
        return 1
    elif analysis.sentiment.polarity == 0:
        return 0
    else:
        return -1


# tweets_df['SA'] = np.array([ analize_sentimentEN(tweet) for tweet in tweets_df['Tweet'] ]) #English Dictionary

tweets_df['SA TextBlob'] = np.array([sentiment(tweet) for tweet in tweets_df['Tweets']])  # Portuguese Dictionary

# Sentiment Analysis Results
pos_tweets = [tweet for index, tweet in enumerate(tweets_df['Tweets']) if tweets_df['SA TextBlob'][index] > 0]
neg_tweets = [tweet for index, tweet in enumerate(tweets_df['Tweets']) if tweets_df['SA TextBlob'][index] < 0]

print("Porcentagem de Tweets Positivos: {}%".format(len(pos_tweets) * 100 / len(tweets_df['Tweets'])))
print("Porcentagem de Tweets Negativos: {}%".format(len(neg_tweets) * 100 / len(tweets_df['Tweets'])))

sentiments = ['Positivos', 'Negativos']
percents = [len(pos_tweets), len(neg_tweets)]

pie_chart = pd.Series(percents, index=sentiments, name='Sentimentos')
pie_chart.plot.pie(fontsize=11, autopct='%.2f', figsize=(6, 6));
print(plt.show())

stopwords = set(STOPWORDS)
new_words = []
with open("C:\\Users\\diego\\Desktop\\data_set\\stopwords_portuguese.txt", 'r') as f:
    [new_words.append(word) for line in f for word in line.split()]

new_stopwords = stopwords.union(new_words)

words = ' '.join(tweets_df['Tweets'])

words_clean = " ".join([word for word in words.split()
                        if 'https' not in word
                        and not word.startswith('@')
                        and word != 'RT'
                        ])

from imageio import imread
import warnings
from PIL import Image
import numpy as np
warnings.simplefilter('ignore')

twitter_mask = np.array(Image.open("brasil_mask_inPixio.jpg"))
print(twitter_mask)
#######################
import numpy as np
import pandas as pd
from PIL import Image
from wordcloud import WordCloud

twitter_mask = np.array(Image.open("brasil_mask_inPixio.jpg"))

#######################################
wc = WordCloud(min_font_size=10,
               max_font_size=300,
               background_color='white',
               mode="RGB",
               stopwords=new_stopwords,
               width=2000,
               height=1000,
               mask=twitter_mask,
               normalize_plurals=True).generate(words_clean)
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.savefig('lula-bolsonaro.png', dpi=300)
from matplotlib import pyplot as plt
plt.savefig('lula-bolsonaro.png')
print(plt.show())

# Time Series
# Tweets per Day
data = tweets_df

data['Date'] = pd.to_datetime(data['Date']).apply(lambda x: x.date())

tlen = pd.Series(data['Date'].value_counts(), index=data['Date'])
tlen.plot(figsize=(16, 4), color='r');
print(plt.show())

# Twitter Setiments Analisys with NLTK¶
# Training Model

vectorizer = CountVectorizer(analyzer="word")
freq_tweets = vectorizer.fit_transform(wordsPT)
modelo = MultinomialNB()
modelo.fit(freq_tweets, wordsPT_sentiments);

# Sentiment Analisys

tweetsarray = []

for tw in tweets_df['Tweets']:
    text = clean_tweet(tw)
    tweetsarray.append(text)

predictionData = vectorizer.transform(tweets_df['Tweets'])
tweets_df['SA NLTK'] = modelo.predict(predictionData)

# Results of Sentiment Analysis
# Sentiment Analysis Results

pos_tweets = [tweet for index, tweet in enumerate(tweets_df['Tweets']) if tweets_df['SA NLTK'][index] > 0]
neg_tweets = [tweet for index, tweet in enumerate(tweets_df['Tweets']) if tweets_df['SA NLTK'][index] < 0]

print("Porcentagem de Tweets Positivos: {}%".format(len(pos_tweets) * 100 / len(tweets_df['Tweets'])))
print("Porcentagem de Tweets Negativos: {}%".format(len(neg_tweets) * 100 / len(tweets_df['Tweets'])))

sentiments = ['Positivos', 'Negativos']
percents = [len(pos_tweets), len(neg_tweets)]

pie_chart = pd.Series(percents, index=sentiments, name='Sentimentos')
pie_chart.plot.pie(fontsize=11, autopct='%.2f', figsize=(6, 6));

# Esse codigo mostra a plot
print(plt.show())

# Map of Tweets
geolocator = Nominatim(user_agent="TweeterSentiments")

latitude = []
longitude = []

for user_location in tweets_df['User Location']:
    try:
        location = geolocator.geocode(user_location)
        latitude.append(location.latitude)
        longitude.append(location.longitude)
    except:
        continue

coordenadas = np.column_stack((latitude, longitude))

mapa = folium.Map(location=[-15.788497, -47.879873], zoom_start=4.)

mapa.add_child(plugins.HeatMap(coordenadas))
mapa.save('lula-bolsonaro.html')
mapa

