# Step 1: Import and display the dataset

import pandas as pd

#Load the data set and repalce the empty rows for NaN valeu
EV_power = pd.read_csv('C:\\Users\\Diego Alves\\Desktop\\Data_sets\\ChargingStations_GB.csv',header=0,encoding = 'unicode_escape')

len(EV_power.columns)

print(EV_power.head())

# iterating the columns
for col in EV_power.columns:
    print(col)