# Step 1: Import and display the dataset

import pandas as pd
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
import statsmodels.api as sm

#Load the data set and repalce the empty rows for NaN valeu
TV = pd.read_csv('C:\\Users\\Diego Alves\\Desktop\\Data_sets\\advertising.csv',header=0,encoding = 'unicode_escape')

#print(pd.get_dummies(TV))

target = TV.sales
predictors = TV.drop('sales', axis=1, inplace=False)

predictors = pd.get_dummies(predictors)

regression_model = sm.OLS(target, predictors)

summary_table = regression_model.fit()

print(summary_table.summary())










