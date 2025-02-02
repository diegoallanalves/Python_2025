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
from adodbapi.adodbapi import Dispatch
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

import os

from termcolor import colored
# to ignore warnings
import warnings

warnings.filterwarnings("ignore")

print(
    '########################################################################################################################################################################################################################################################################################################################################')

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
    Regional_Data = pd.read_sql_query(sa.text("SELECT TOP(100000) [MVRISPostcode], [Make], [Range], [Colour], [Count of Registrations]  FROM [Parc].[dbo].[DataShop] WHERE [Make] ='Audi'"), conn)

print('Saved file')
Regional_Data.to_excel(
    r'C:\\Users\\alvesd\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\python\\21_Car_Predict_Methods_with_Ensemble\\Data\\Car_Data.xlsx',
    index=False, sheet_name='Audi')

print(
    '########################################################################################################################################################################################################################################################################################################################################')

# Open the Parc_Data:
print('Step : Parc_Data')
Parc_Data = pd.read_excel(
    'C:\\Users\\alvesd\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\python\\21_Car_Predict_Methods_with_Ensemble\\Data\\Car_Data.xlsx',
    index_col=False)

# Open the 1_Terminated_Postcodes_Data:
print('Step : Geo_Data')
Geo_Data = pd.read_excel(
    'C:\\Users\\alvesd\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\python\\16_Geo_Data_Automation\\ONS Data\\1_Final_Geo_Data.xlsx',
    index_col=False)
print(
    '########################################################################################################################################################################################################################################################################################################################################')

# print the list using tolist()
print("The column headers :")
print(Geo_Data.columns.tolist())
print(Parc_Data.columns.tolist())

print(
    '########################################################################################################################################################################################################################################################################################################################################')
Parc_Data = Parc_Data.rename(columns={'MVRISPostcode': 'AlternativePostcode'})

print('Step xx: Vlook up ONSDistrict')
# Vlook up ONSDistrict:
Parc_District_by_Color = pd.merge(Geo_Data,
                                  Parc_Data[['AlternativePostcode', 'Make', 'Range', 'Colour', 'Count of Registrations']],
                                  on='AlternativePostcode', how='left')

print(Parc_District_by_Color.columns.tolist())

Parc_District_by_Color.dropna(subset=['Colour'], inplace=True)

print(
    '########################################################################################################################################################################################################################################################################################################################################')
print('Step xx: Save file')
Parc_District_by_Color.to_excel(r'C:\\Users\\alvesd\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\python\\21_Car_Predict_Methods_with_Ensemble\\Data\\Parc_District_Color.xlsx', index=False, header=True, sheet_name='Audi')

print(
    '########################################################################################################################################################################################################################################################################################################################################')
print('Step xx: Match rows to text, format the cells for better look')
excel = Dispatch('Excel.Application')
wb = excel.Workbooks.Open(
    "C:\\Users\\alvesd\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\python\\21_Car_Predict_Methods_with_Ensemble\\Data\\Parc_District_Color.xlsx")

# Activate second sheet
excel.Worksheets(1).Activate()

# Autofit column in active sheet
excel.ActiveSheet.Columns.AutoFit()

# Or simply save changes in a current file
wb.Save()

wb.Close()

print(
    '########################################################################################################################################################################################################################################################################################################################################')
print('Done up to here!!!')
