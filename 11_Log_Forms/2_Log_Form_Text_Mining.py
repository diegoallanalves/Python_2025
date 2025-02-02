## Use the repetitive errors brought up by our customers as a reference to improve our services.

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the data
df = pd.read_excel (f'F:\\AIS\\Enquiries and Pricing\\Enquiries Log - 2022\\Dashboard\\Test_SDI Logs and Enquiries Forum.xlsx', 'Enquiry Sheet')

df = df.dropna(subset=['Question/ requests (Free text)'])

corpus = list(df['Question/ requests (Free text)'].values)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

threshold = 0.4

for x in range(0,X.shape[0]):
  for y in range(x,X.shape[0]):
    if(x!=y):
      if(cosine_similarity(X[x],X[y])>threshold):
        print(df["Email address"][x],":",corpus[x])
        print(df["Email address"][y],":",corpus[y])
        print("Cosine similarity:",cosine_similarity(X[x],X[y]))
        print()


#############################################################################################################################################################
import textdistance

rank = (
    df.assign(
        match=df["Question/ requests (Free text)"].map(
            lambda x: max(
                [textdistance.jaro_winkler(x, text) for text in df["Question/ requests (Free text)"]],
                key=lambda x: x if x != 1 else 0,
            )
        )
    )
    .sort_values(by="match")
    .reset_index(drop=True)
)

# sort the vowels
rank = rank.sort_values('match', ascending=False)

# Export the results to Excel
rank.to_excel(r'F:\\AIS\\Enquiries and Pricing\\Enquiries Log - 2022\\Dashboard\\Rank Results.xlsx', index = False)

#############################################################################################################################################################
## Sentence similarity requests

# Load the data
df1 = pd.read_excel (f'F:\\AIS\\Enquiries and Pricing\\Enquiries Log - 2022\\Dashboard\\Test_SDI Logs and Enquiries Forum.xlsx', 'Enquiry Sheet')