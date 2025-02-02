import webbrowser

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib_inline.config import InlineBackend

np.random.seed(0)
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler, LabelEncoder

scale = StandardScaler()

# Load the data set and repalce the empty rows for NaN valeu
parc = pd.read_excel(
    "C:\\Users\\alvesd\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\python\\33_Sales_Target_Region_Vehicle_Seats\\Parc_Data.xlsx")

# Print Headers
print(parc.head(5))
# Print columns headers:
column_headers = list(parc.columns.values)
print("The Column Header :", column_headers)

# Turn Texts to numbers
le = LabelEncoder()
parc = parc[parc.columns[:]].apply(le.fit_transform)

# Save numeric data to file:
parc.to_excel(
    'C:\\Users\\alvesd\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\python\\33_Sales_Target_Region_Vehicle_Seats\\Parc_Numeric_Data.xlsx',
    index=False)

# Solihull Prediction
predictors = parc[['Number of Seats', 'Body Style']]
target = parc['Registrations']

predictors = pd.get_dummies(predictors)
regression_model = sm.OLS(target, predictors)

summary_table = regression_model.fit()

print(summary_table.summary())

########################################################################################################################################################

# Clustering Geolocation Data in Python using DBSCAN and K-Means
import os

print(os.path.abspath("."))
from collections import defaultdict
from ipywidgets import interactive
import hdbscan
import folium
import re
import matplotlib
# %matplotlib inline
# %config InlineBackend.figure_format = 'svg'
import matplotlib.pyplot as plt

plt.style.use('ggplot')
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
import folium
from folium import plugins
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML

# Load the data set and repalce the empty rows for NaN valeu
parc_cluster = pd.read_excel(
    "C:\\Users\\alvesd\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\python\\33_Sales_Target_Region_Vehicle_Seats\\Parc_Data.xlsx")

# Define colours:
cols = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
        '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075',
        '#808080'] * 10

print(parc_cluster.head())

# Drop the Null and duplicates
print(parc_cluster.duplicated(subset=['longitude', 'latitude']).values.any())

print(parc_cluster.isna().values.any())

print(f'Before (Nulls and Duplicates) \t:\tParc_cluster.shape = {parc_cluster.shape}')
parc_cluster.dropna(inplace=True)
parc_cluster.drop_duplicates(subset=['longitude', 'latitude'], keep='first', inplace=True)
print(f'After (Nulls and Duplicates) \t:\tParc_cluster.shape = {parc_cluster.shape}')

# Plot the points:
X = np.array(parc_cluster[['longitude', 'latitude']], dtype='float64')
plt.scatter(X[:, 0], X[:, 1], alpha=0.2, s=50)
plt.show()

# Using Folium Visualize Geographical Data
# Folium
# Folium makes it easy to visualize data that’s been manipulated in Python on an interactive leaflet map. It enables both the binding of data to a map for
# choropleth visualizations as well as passing rich vector/raster/HTML visualizations as markers on the map.

# m = folium.Map(location=[parc_cluster['latitude'].mean(), parc_cluster['longitude'].mean()], zoom_start=9,
#               tiles='Stamen Toner')

m = folium.Map(location=[parc_cluster['latitude'].mean(), parc_cluster['longitude'].mean()], zoom_start=7)

for _, row in parc_cluster.iterrows():
    folium.CircleMarker(
        location=[row.latitude, row.longitude],
        radius=5,
        popup=re.sub(r'[^a-zA-Z]+', '', row.Make),
        color='#1787FE',
        fill=True,
        fill_color='#1787FE').add_to(m)
m.save('seats.html')

# Clustering Strength
X_blobs, _ = make_blobs(n_samples=2654, centers=10,
                        n_features=2, cluster_std=0.5, random_state=4)
plt.scatter(X_blobs[:, 0], X_blobs[:, 1], alpha=0.2)
plt.show()

# Load the data set and repalce the empty rows for NaN valeu
Parc_Numeric_Data = pd.read_excel(
    "C:\\Users\\alvesd\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\python\\33_Sales_Target_Region_Vehicle_Seats\\Parc_Numeric_Data.xlsx")

# Drop the Null and duplicates
print(Parc_Numeric_Data.duplicated(subset=['longitude', 'latitude']).values.any())

print(Parc_Numeric_Data.isna().values.any())

print(f'Before (Nulls and Duplicates) \t:\tParc_cluster.shape = {Parc_Numeric_Data.shape}')
Parc_Numeric_Data.dropna(inplace=True)
Parc_Numeric_Data.drop_duplicates(subset=['longitude', 'latitude'], keep='first', inplace=True)
print(f'After (Nulls and Duplicates) \t:\tParc_cluster.shape = {Parc_Numeric_Data.shape}')

class_predictions = Parc_Numeric_Data['Company/Private']
unique_clusters = np.unique(class_predictions)
for unique_clusters in unique_clusters:
    X = X_blobs[class_predictions == unique_clusters]
    plt.scatter(X[:, 0], X[:, 1], alpha=0.2, c=cols[unique_clusters])
    plt.show()

print(silhouette_score(X_blobs, class_predictions))

class_predictions = Parc_Numeric_Data['Body Style']
unique_clusters = np.unique(class_predictions)
for unique_clusters in unique_clusters:
    X = X_blobs[class_predictions == unique_clusters]
    plt.scatter(X[:, 0], X[:, 1], alpha=0.2, c=cols[unique_clusters])

print(silhouette_score(X_blobs, class_predictions))

##################################################################

