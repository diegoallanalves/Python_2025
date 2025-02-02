# Step 1: Import and display the dataset

import pandas as pd
import numpy as np

EV_power = pd.read_csv('C:\\Users\\Diego Alves\\Desktop\\Data_sets\\ChargingStations_GB.csv',header=0,encoding = 'unicode_escape')

print(EV_power)

# Step 2: iterating the columns
for col in EV_power.columns:
    print(col)

# Step 3: select the attributes for the analysis and change to numerical data, if is necessary.
Ev_power_attributes = pd.get_dummies(EV_power, columns = ['UsageType_IsAccessKeyRequired', 'UsageType_IsMembershipRequired', 'UsageType_IsPayAtLocation'], drop_first = True)

# iterating the columns
for col in Ev_power_attributes.columns:
    print(col)

# Import library
import matplotlib

from matplotlib import pyplot as plt

Ev_power_attributes['UsageType_IsAccessKeyRequired_True'].value_counts().sort_index().plot.barh()
plt.show()

Ev_power_attributes['UsageType_IsAccessKeyRequired_True'].plot.hist()
plt.show()

Ev_power_attributes['UsageType_IsAccessKeyRequired_True'].value_counts().sort_index().plot.bar()
plt.show()

#
import matplotlib.pyplot as plt

# create dummy variable them group by that
# set the legend to false because we'll fix it later
EV_power.assign(dummy = 1).groupby(
  ['dummy','UsageType_IsAccessKeyRequired']
).size().to_frame().unstack().plot(kind='bar',stacked=True,legend=False)

plt.title('UsageType_IsAccessKeyRequired')

# other it'll show up as 'dummy'
plt.xlabel('state')

# disable ticks in the x axis
plt.xticks([])

# fix the legend
current_handles, _ = plt.gca().get_legend_handles_labels()
reversed_handles = reversed(current_handles)

labels = reversed(EV_power['UsageType_IsAccessKeyRequired'].unique())

plt.legend(reversed_handles,labels,loc='lower right')
plt.show()
