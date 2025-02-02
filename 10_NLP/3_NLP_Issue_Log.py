#Code source: https://www.kaggle.com/code/hardikjain10/email-sentiment-analysis
###################################################################################################
#Loading libraries
import numpy as np # provides a high-performance multidimensional array and tools for its manipulation
import pandas as pd # for data munging, it contains manipulation tools designed to make data analysis fast and easy
import re # Regular Expressions - useful for extracting information from text
import nltk # Natural Language Tool Kit for symbolic and statistical natural language processing
import spacy # processing and understanding large volumes of text
import string # String module contains some constants, utility function, and classes for string manipulation
import email
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')

# For viz
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
from collections import Counter

from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
pd.options.mode.chained_assignment = None

import numpy as np
import pandas as pd

print('###############################################################################################################')
df = pd.read_excel('C:\\Users\\alvesd\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\python\\10_NLP\Output\Data\\SMMT Data Intelligence enquiries log.xlsx', index_col='Email address')

print('###############################################################################################################')
print(df.head())
print(df.info())

print('###############################################################################################################')
# create list of email objects
emails = list(map(email.parser.Parser().parsestr,df['Request details']))

print('###############################################################################################################')
# extract headings such as subject, from, to etc..
headings = emails[0].keys()

print('###############################################################################################################')
# Goes through each email and grabs info for each key
# doc['From'] grabs who sent email in all emails
for key in headings:
    df[key] = [doc[key] for doc in emails]

print('###############################################################################################################')
print(df.head())

print('###############################################################################################################')
def clean_column(data):
    if data is not None:
        stopwords_list = stopwords.words('english')
        # exclusions = ['RE:', 'Re:', 're:']
        # exclusions = '|'.join(exclusions)
        data = data.lower()
        data = re.sub('re:', '', data)
        data = re.sub('-', '', data)
        data = re.sub('_', '', data)
        # Remove data between square brackets
        data = re.sub('\[[^]]*\]', '', data)
        # removes punctuation
        data = re.sub(r'[^\w\s]', '', data)
        data = re.sub(r'\n', ' ', data)
        data = re.sub(r'[0-9]+', '', data)
        # strip html
        p = re.compile(r'<.*?>')
        data = re.sub(r"\'ve", " have ", data)
        data = re.sub(r"can't", "cannot ", data)
        data = re.sub(r"n't", " not ", data)
        data = re.sub(r"I'm", "I am", data)
        data = re.sub(r" m ", " am ", data)
        data = re.sub(r"\'re", " are ", data)
        data = re.sub(r"\'d", " would ", data)
        data = re.sub(r"\'ll", " will ", data)
        data = re.sub('forwarded by phillip k allenhouect on    pm', '', data)
        data = re.sub(r"httpitcappscorpenroncomsrrsauthemaillinkaspidpage", "", data)

        data = p.sub('', data)
        if 'forwarded by:' in data:
            data = data.split('subject')[1]
        data = data.strip()
        return data
    return 'No Subject'

df['Solution'] = df.rename(columns = {'Solution (comment)':'Solution'}, inplace = True)

print('###############################################################################################################')
df['Request details new'] = df['Request details'].apply(clean_column)
df['Solution'] = df['Solution'].apply(clean_column)

print('###############################################################################################################')
print(df['Request details new'].head(5))

print('###############################################################################################################')
print(df['Solution'].head(5))

print('###############################################################################################################')

# Drop duplicates
df.drop_duplicates()
print(" Shape of dataframe after dropping duplicates: ", df.shape)

print('###############################################################################################################')
# Null values
null = df.isnull().sum().sort_values(ascending=False)
total = df.shape[0]
percent_missing = (df.isnull().sum() / total).sort_values(ascending=False)

missing_data = pd.concat([null, percent_missing], axis=1, keys=['Total missing', 'Percent missing'])

missing_data.reset_index(inplace=True)
missing_data = missing_data.rename(columns={"index": " column name"})

print("Null Values in each column:\n", missing_data)

print('###############################################################################################################')
import vaderSentiment
# calling SentimentIntensityAnalyzer object
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()

# Using polarity scores for knowing the polarity of each text
def sentiment_analyzer_score(sentence):
    score = analyser.polarity_scores(sentence)
    print("{:-<40} {}".format(sentence, str(score)))

#testing the function
email  = "I like dogs"
email2 = "I like hotdogs"
email3 = "I dont like hotdogs"
print (sentiment_analyzer_score(email))
print (sentiment_analyzer_score(email2))
print (sentiment_analyzer_score(email3))

print('###############################################################################################################')
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
words_descriptions = df['Request details new'].apply(tokenizer.tokenize)
print(words_descriptions.head())

all_words = [word for tokens in words_descriptions for word in tokens]
df['description_lengths']= [len(tokens) for tokens in words_descriptions]
VOCAB = sorted(list(set(all_words)))
print("%s words total, with a vocabulary size of %s" % (len(all_words), len(VOCAB)))

# Checking most common words
from collections import Counter
count_all_words = Counter(all_words)
print(count_all_words.most_common(100))

print('###############################################################################################################')
from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)
to_add = ['FW', 'ga', 'httpitcappscorpenroncomsrrsauthemaillinkaspidpage', 'cc', 'aa', 'aaa', 'aaaa',
         'hou', 'cc', 'etc', 'subject', 'pm']

for i in to_add:
    stopwords.add(i)

#Visualise Email Subject

wordcloud = WordCloud(
                          collocations = False,
                          width=1600, height=800,
                          background_color='white',
                          stopwords=stopwords,
                          max_words=150,
                          #max_font_size=40,
                          random_state=42
                         ).generate(' '.join(df['Request details new'])) # can't pass a series, needs to be strings and function computes frequencies
print(wordcloud)
plt.figure(figsize=(9,8))
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

print('###############################################################################################################')
df['scores'] = df['Request details new'].apply(lambda review: analyser.polarity_scores(review))
print(df.head())

df['compound']  = df['scores'].apply(lambda score_dict: score_dict['compound'])

print(df.head())

print('###############################################################################################################')
def Sentimnt(x):
    if x>= 0.05:
        return "Positive"
    elif x<= -0.05:
        return "Negative"
    else:
        return "Neutral"
#df['Sentiment'] = df['compound'].apply(lambda c: 'positive' if c >=0.00  else 'negative')
df['Sentiment'] = df['compound'].apply(Sentimnt)

print(df.head())

print('###############################################################################################################')
var1 = df.groupby('Sentiment').count()['Solution'].reset_index().sort_values(by='Solution',ascending=False)
fig = go.Figure(go.Funnelarea(
    text =var1.Sentiment,
    values = var1.Solution,
    title = {"position": "top center", "text": "Funnel-Chart of Sentiment Distribution"}
    ))
fig.show()
plt.show()

print('###############################################################################################################')
df['temp_list'] = df['Request details new'].apply(lambda x:str(x).split())
top = Counter([item for sublist in df['temp_list'] for item in sublist])
temp = pd.DataFrame(top.most_common(20))
temp.columns = ['Common_words','count']
#temp.style.background_gradient(cmap='Blues')

print('###############################################################################################################')
fig = px.bar(temp, x="count", y="Common_words", title='Commmon Words in Selected Text', orientation='h',
             width=700, height=700,color='Common_words')
fig.show()
plt.show()

# iterating the columns
for col in df.columns:
    print(col)

print('###############################################################################################################')
comment_words = ''
stopwords = set(STOPWORDS)

df_positive = df[df["Sentiment"] == "Positive"]
# iterate through the csv file
for val in df_positive["Request details new"]:

    # typecaste each val to string
    val = str(val)

    # split the value
    tokens = val.split()

    # Converts each token into lowercase
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()

    comment_words += " ".join(tokens) + " "

wordcloud = WordCloud(width=800, height=800,
                      background_color='white',
                      stopwords=stopwords,
                      min_font_size=10).generate(comment_words)

# plot the WordCloud image
plt.figure(figsize=(8, 8), facecolor="green")
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)

plt.show()

print('###############################################################################################################')
comment_words = ''
stopwords = set(STOPWORDS)

df_positive = df[df["Sentiment"] == "Negative"]
# iterate through the csv file
for val in df_positive["Request details new"]:

    # typecaste each val to string
    val = str(val)

    # split the value
    tokens = val.split()

    # Converts each token into lowercase
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()

    comment_words += " ".join(tokens) + " "

wordcloud = WordCloud(width=800, height=800,
                      background_color='white',
                      stopwords=stopwords,
                      min_font_size=10).generate(comment_words)

# plot the WordCloud image
plt.figure(figsize=(8, 8), facecolor="red")
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)

plt.show()

print('###############################################################################################################')
import warnings
warnings.filterwarnings('ignore')
from textblob import TextBlob, Word, Blobber

email = "I would love to meet you again"
TextBlob(email).sentiment

# Applying on dataset
df['TB_score']= df["Request details new"].apply(lambda x: TextBlob(x).sentiment)
print(df.head())

print('###############################################################################################################')
df['TB_sentiment'] = df['Request details new'].apply(lambda x: TextBlob(x).sentiment[0])
print(df.head())

print('###############################################################################################################')
from nrclex import NRCLex
email = NRCLex('Good work to the team')
#Return affect dictionary
print(email.affect_dict)
#Return raw emotional counts
print("\n",email.raw_emotion_scores)
#Return highest emotions
print("\n", email.top_emotions)
#Return affect frequencies
print("\n",email.affect_frequencies)

print('###############################################################################################################')
def emotion(x):
    text = NRCLex(x)
    if text.top_emotions[0][1] == 0.0:
        return "No emotion"
    else:
        return text.top_emotions[0][0]
df['Emotion'] = df['Request details new'].apply(emotion)
print(df.head())

print('###############################################################################################################')
import matplotlib.pyplot as plt
from matplotlib import cm
from math import log10

df_chart = df[df.Emotion != "No emotion"]
labels = df_chart.Emotion.value_counts().index.tolist()
data = df_chart.Emotion.value_counts()
#number of data points
n = len(data)
#find max value for full ring
k = 10 ** int(log10(max(data)))
m = k * (1 + max(data) // k)

#radius of donut chart
r = 1.5
#calculate width of each ring
w = r / n

#create colors along a chosen colormap
colors = [cm.terrain(i / n) for i in range(n)]

#create figure, axis
fig, ax = plt.subplots()
ax.axis("equal")

#create rings of donut chart
for i in range(n):
    #hide labels in segments with textprops: alpha = 0 - transparent, alpha = 1 - visible
    innerring, _ = ax.pie([m - data[i], data[i]], radius = r - i * w, startangle = 90, labels = ["", labels[i]], labeldistance = 1 - 1 / (1.5 * (n - i)), textprops = {"alpha": 0}, colors = ["white", colors[i]])
    plt.setp(innerring, width = w, edgecolor = "white")

plt.legend()
plt.show()