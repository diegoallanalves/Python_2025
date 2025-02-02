import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
np.random.seed(0)
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

#Load the data set and repalce the empty rows for NaN valeu
EV = pd.read_excel('C:\\Users\\alvesd\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\data_set\\project_data.xlsx')

#Solihull Prediction
predictors = EV[['Solihull_Population', 'Year_Pay_Workers Solihull', 'PHEVs Registered in Solihull']]
target = EV['Solihull_charging_points']

predictors = pd.get_dummies(predictors)
regression_model = sm.OLS(target, predictors)

summary_table = regression_model.fit()

print(summary_table.summary())


