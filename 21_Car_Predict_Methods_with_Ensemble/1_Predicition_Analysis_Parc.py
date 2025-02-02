import os
# ignore warnings
import warnings
warnings.filterwarnings("ignore")

path = r'C:\\Users\\alvesd\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\python\\21_Car_Predict_Methods_with_Ensemble\\Data\\'
os.chdir(path)
for file in os.listdir(path):
    if file.endswith('.xlsx') or file.endswith('.xls'):
        print(file)
        os.remove(file)

from sqlalchemy.engine import URL
from sqlalchemy.sql import text
from sqlalchemy import create_engine
import pandas as pd
import sqlalchemy as sa

connection_string = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=GPS-SRV20;DATABASE=Parc;UID=SMMT\\alvesd;PWD=;Trusted_connection=yes"
connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": connection_string})

engine = create_engine(connection_url)

with engine.begin() as conn:
    Regional_Data = pd.read_sql_query(sa.text("SELECT TOP (1000) * FROM [Parc].[dbo].[DataShop] WHERE [Make] ='Audi'"), conn)

print('Saved file')
Regional_Data.to_excel(
    r'C:\\Users\\alvesd\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\python\\21_Car_Predict_Methods_with_Ensemble\\Data\\Car_Data.xlsx',
    index=False)


# https://www.kaggle.com/code/eisgandar/car-prices-predict-with-ensemble-methods
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import scipy.stats as st
import math
import datetime
import missingno as msno
from scipy.stats import norm, skew
from sklearn import metrics
from collections import Counter

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score
from sklearn import model_selection
from sklearn.pipeline import make_pipeline

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from mlxtend.regressor import StackingCVRegressor

from termcolor import colored
# to ignore warnings
import warnings
warnings.filterwarnings("ignore")

#to see model hyperparameters
from sklearn import set_config
set_config(print_changed_only = False)

# to show all columns
pd.set_option('display.max_columns', 15)

car = pd.read_excel("C:\\Users\\alvesd\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\python\\21_Car_Predict_Methods_with_Ensemble\\Data\\Car_Data.xlsx")
df = car.copy()
df.head().style.background_gradient(cmap = "autumn")

df['Miles_Driven'] = np.random.choice([1, 5, 9, 11, 15, 20], df.shape[0])
df['Price'] = np.random.choice([50, 120, 167, 1344, 10010,  1550, 20500], df.shape[0])
#df.head

print("The number of rows in train data is {0}, and the number of columns in train data is {1}".
      format(df.shape[0], df.shape[1]))

print(df.info())

#make dtypes of some variables 'category'

categoric_columns = ["Fuel Type", "Transmission", "Make", "Company_Private"]
for column in categoric_columns:
    df[column] = df[column].astype("category")

#create 'car_brand_name' feature from 'name' feature

df["car_brand_name"] = df["Make"].str.extract('([^\s]+)')
df["car_brand_name"] = df["car_brand_name"].astype("category")

#create 'car_age' feature from 'year' column

df["car_age"] = (datetime.datetime.now().year) - (df["Year of 1st Reg"])

print(df.head().style.background_gradient(cmap = "summer"))

print(df.info())

# check whether there are null values in the dataset
print(df.isnull().sum())

print(df.describe().T.style.background_gradient(cmap = "viridis"))

print(df.describe(include = "category").T)

print(msno.matrix(df))

# visualize missing values with seaborn (distplot)

plt.figure(figsize = (10, 5))
sns.displot(
    data = df.isna().melt(value_name = "missing"),
    y = "variable",
    hue = "missing",
    multiple = "fill",
    aspect = 2
)

# visualize missing values with seaborn (heatmap)

plt.figure(figsize = (15, 5))
sns.heatmap(df.isna().transpose(),
            cmap = "YlGnBu",
            cbar_kws = {'label': 'Missing Data'})

# fill null values with median (numeric) and frequent values (categoric)

numeric_data = [column for column in df.select_dtypes(["int", "float"])]
categoric_data = [column for column in df.select_dtypes(exclude=["int", "float"])]

for col in numeric_data:
    df[col].fillna(df[col].median(), inplace=True)

# replace missing values in each categorical column with the most frequent value
for col in categoric_data:
    df[col].fillna(df[col].value_counts().index[0], inplace=True)

#check null values again

print(df.isnull().sum().sum())

#get class frequencies of some variables

print("Class frequencies of 'Fuel Type' variable: \n\n", df["Fuel Type"].value_counts())

print("_____________________________________________________________________________________")

print("Class frequencies of 'Country of Origin' variable: \n\n", df["Country of Origin"].value_counts())

print("_____________________________________________________________________________________")

print("Class frequencies of 'Company_Private' variable: \n\n", df["Company_Private"].value_counts())

print("_____________________________________________________________________________________")

print("Class frequencies of 'car_brand_name' variable: \n\n", df["car_brand_name"].value_counts())

#check correlation between the variables of dataset

#df.corr().style.background_gradient(cmap = "copper")

fig, axes = plt.subplots(1, 3, figsize = (30, 7))

sns.boxplot(ax = axes[0], x = "Price", data = df, width = 0.5, fliersize = 3, linewidth = 1);
sns.boxplot(ax = axes[1], x = "Miles_Driven", data = df, width = 0.5, fliersize = 3, linewidth = 1);
sns.boxplot(ax = axes[2], x = "Number Previous Keepers", data = df, width = 0.5, fliersize = 3, linewidth = 1);

fig, axes = plt.subplots(2, 2, figsize = (30, 20))
axes = axes.flatten()

sns.boxplot(ax = axes[0], x = "Engine CC", data = df, width = 0.5, fliersize = 3, linewidth = 1);
sns.boxplot(ax = axes[1], x = "Power BHP", data = df, width = 0.5, fliersize = 3, linewidth = 1);
sns.boxplot(ax = axes[2], x = "Number of Seats", data = df, width = 0.5, fliersize = 3, linewidth = 1);
sns.boxplot(ax = axes[3], x = "car_age", data = df, width = 0.5, fliersize = 3, linewidth = 1);

df.hist(figsize = (30, 25), bins = 30, legend = False)
plt.show()

sns.catplot(x = "Company_Private",
            y = "Price",
            kind = "boxen",
            height = 7,
            aspect = 1,
            color = "#671A76",
            data = df).set(title = "Country of Origin by Price");

sns.catplot(x = "Fuel Type",
            y = "Price",
            kind = "strip",
            hue = "Company_Private",
            height = 7,
            aspect = 1.4,
            color = "#661E1D",
            data = df).set(title = "Company_Privates by Fuel Type");

sns.catplot(x = "Make",
            y = "Price",
            kind = "boxen",
            height = 7,
            aspect = 1.37,
            color = "#F0F312",
            data = df).set(title = "car_age by Number Previous Keepers");

price = df["Price"]
brand = df["Make"]
dff = pd.concat([price, brand], axis = 1)
f, ax = plt.subplots(figsize = (50, 30))
fig = sns.boxplot(x=dff["Make"], y=dff["Price"]);

fig, axes = plt.subplots(1, 5, figsize = (50, 10))

sns.barplot(ax = axes[0], x = df["Country of Origin"].value_counts().index, y = df["Country of Origin"].value_counts(),
            saturation = 1).set(title = "Frequency of classes of the 'Country of Origin' variable");

sns.barplot(ax = axes[1], x = df["Company_Private"].value_counts().index, y = df["Company_Private"].value_counts(),
            saturation = 1).set(title = "Frequency of classes of the 'Company_Private' variable");

sns.barplot(ax = axes[2], x = df["Fuel Type"].value_counts().index, y = df["Fuel Type"].value_counts(),
            saturation = 1).set(title = "Frequency of classes of the 'Fuel Type' variable");

sns.barplot(ax = axes[3], x = df["Transmission"].value_counts().index, y = df["Transmission"].value_counts(),
            saturation = 1).set(title = "Frequency of classes of the 'Transmission' variable");

sns.barplot(ax = axes[4], x = df["Make"].value_counts().index, y = df["Make"].value_counts(),
            saturation = 1).set(title = "Frequency of classes of the 'Make' variable");

plt.figure(figsize = (15, 8))
sns.barplot(x = "Transmission", y = "Price", hue = "Make", data = df, saturation = 1);

plt.figure(figsize = (15, 8))
sns.barplot(x = "Fuel Type", y = "Price", hue = "Company_Private", data = df, saturation = 1);

plt.figure(figsize = [8, 8], clear = True, facecolor = "#FFFFFF")
df["Fuel Type"].value_counts().plot.pie(explode = None, autopct='%1.3f%%', shadow = True);

plt.figure(figsize = [8, 8], clear = True, facecolor = "#FFFFFF")
df["Make"].value_counts().plot.pie(explode = None, autopct='%1.3f%%', shadow = True);

sns.displot(data = df, x = "Price", hue = "Make", kind = "kde", height = 6,
            aspect = 1.3, clip=(0, None), palette="ch:rot=-.25, hue = 2, light=.20"
).set(title = "density of the classes of 'owner' variable by 'selling price' ");

sns.displot(
    data = df, x = "Price", hue = "Make",
    kind = "ecdf", height = 5, aspect = 1.8).set(title =  "density relationship between 'Number Previous Keepers' and 'Fuel Type' variables");

sns.displot(
    data = df, x = "car_age", hue = "Fuel Type",
    kind = "kde", height = 5, aspect = 1.8, multiple="fill").set(title = "density relationship between 'Year of 1st Reg' and 'Fuel Type' variables");

sns.displot(
    data = df, x = "Miles_Driven", hue = "Make",
    kind = "kde", height = 5, aspect = 1.8, multiple="fill").set(title = "density relationship between 'Number Previous Keepers' and 'Company_Private' variables");

fig = px.histogram(df, x = "car_age",
                   y = "Price",
                   marginal = None, text_auto = True,
                   color = "Make", hover_data  = df.columns, width = 850, height = 500)
fig.show()

fig = px.histogram(df, x = "Engine CC",
                   y = "Miles_Driven",
                   marginal = None, text_auto = True,
                   color = "Fuel Type", hover_data  = df.columns, width = 850, height = 500)
fig.show()

fig = px.histogram(df, x = "car_age",
                   y = "Miles_Driven",
                   marginal = None, text_auto = True,
                   color = "Make", hover_data  = df.columns, width = 850, height = 500)
fig.show()

fig = px.density_heatmap(df, x = "Power BHP", y = "Price", z = "Miles_Driven",
                        color_continuous_scale = "deep", text_auto = True,
                        title = "Density heatmap between variables")
fig.show()

fig = px.density_heatmap(df, x = "Number of Seats", y = "Engine CC", z = "Price", color_continuous_scale = "portland",
                         text_auto = True, title = "Density heatmap between variables")
fig.show()

fig, ax = plt.subplots(figsize = (15, 5))
sale_price = list()
for sp in df["Price"].values:
    sale_price.append(sp)
sale_price = pd.Series(sale_price)
sale_price.plot(kind = "line", colormap = "winter").set_title("Sale prices of the cars");

sns.pairplot(df, hue = "Make", diag_kind = "hist", corner = True);

fig, axes = plt.subplots(2, 3, figsize=(20, 14))
axes = axes.flatten()

sns.regplot(ax = axes[0], x = "Miles_Driven", y = "Price", data = df);
sns.regplot(ax = axes[1], x = "Engine CC", y = "Price", data = df);
sns.regplot(ax = axes[2], x = "Power BHP", y = "Price", data = df);
sns.regplot(ax = axes[3], x = "Number of Seats", y = "Price", data = df);
sns.regplot(ax = axes[4], x = "car_age", y = "Price", data = df);
sns.regplot(ax = axes[5], x = "Number of Seats", y = "Price", data = df);

plt.figure(figsize = [40, 20], facecolor = "#F7F4F4")
#sns.heatmap(df.corr(), annot = True, linewidths = 2, linecolor = "white", cmap = "viridis");

#df.corr().style.background_gradient(cmap = "binary")

print("Basic descriptive statistics of the target variable - 'Price': \n\n",
      df["Price"].describe())

print("Skewness of target variable: ", df["Price"].skew())
print("Kurtosis of target variable: ", df["Price"].kurt())

sns.set(rc = {"figure.figsize" : (12, 7)})
sns.distplot(df["Price"], bins = 100, color = "red");

df["Price"] = np.log1p(df["Price"])
df["Price"].head(n = 10)

sns.distplot(df["Price"], fit = norm, color = "green");

# get skewness of other numeric variables

numeric_data = [column for column in df.select_dtypes(["int", "float"])]
for col in numeric_data:
    print("Skewness of", col, "variable is:", df[col].skew())

# fix skewness  of them with 'log1p' function

for c in numeric_data:
    df[c] = np.log1p(df[c])

# select dependent variable (label)

y = df["Price"]

# select independent variable (estimator)
x = df.drop("Price", axis = 1)

#encode the variables of the dataset
x = pd.get_dummies(x, drop_first = True)

y.shape, x.shape

x.head(n = 7).style.background_gradient(cmap = "plasma")

# Split the dataset into x_train (y_train) and x_test (y_test) sets

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size = 0.20,
                                                    shuffle = True,
                                                    random_state = 1)
print(x_train.shape)
print(x_test.shape)

rob_scaler = RobustScaler()
rob_scaler.fit(x_train)
x_train = rob_scaler.transform(x_train)
x_test = rob_scaler.transform(x_test)

k_fold = KFold(n_splits = 10, random_state = 11, shuffle = True)

def cv_rmse(model, X = x_train):
    rmse = np.sqrt(-cross_val_score(model, x_train, y_train, scoring = "neg_mean_squared_error", cv = k_fold))
    return rmse

def rmsle(y, y_pred):
    rmsle = np.sqrt(mean_squared_log_error(y, y_pred, squared = False))
    return rmsle

xgb = XGBRegressor(n_estimators = 1000, random_state = 1)
lgbm = LGBMRegressor(n_estimators = 1000, random_state = 1)
gbr = GradientBoostingRegressor(n_estimators = 1000, random_state = 11)
rf = RandomForestRegressor(n_estimators = 1000, random_state = 1)
svr = SVR(C = 20)
lasso = LassoCV(alphas = [1e-10, 1e-8, 1e-7, 1e-5, 1e-2, 9e-4, 9e-3,
                                                        5e-4, 3e-4, 1e-4, 1e-3, 1e-2, 0.1,
                                                        0.3, 0.6, 1, 3, 5, 7, 14, 18, 25, 30,
                                                        45, 50, 70, 90], n_jobs = -1, cv = k_fold)

stacked = StackingCVRegressor(regressors = (xgb, lgbm, svr, lasso, gbr, rf),
                              meta_regressor = xgb, use_features_in_secondary = True)

#fit the stacked model

stacked_model = stacked.fit(np.array(x_train), np.array(y_train))

#RMSLE score of the stacked model on full TRAIN data

stacked_score_train = rmsle(y_train, stacked_model.predict(x_train))
print("RMSLE score of stacked models on full train data:", stacked_score_train)

#RMSLE score of the stacked model on full TEST data

stacked_score_test = rmsle(y_test, stacked_model.predict(x_test))
print("RMSLE score of stacked models on full test data:", stacked_score_test)

y_pred = np.expm1(stacked_model.predict(x_test))
y_pred[0:5]

print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R-squared: {}'.format(metrics.r2_score(y_test, y_pred)))

plt.scatter(y_test, y_pred);
plt.xlabel("actual")
plt.ylabel("prediction")