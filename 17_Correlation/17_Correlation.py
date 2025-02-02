import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

print(
    '########################################################################################################################################################################################################################################################################################################################################')
print('Create a TEXT file for printing the report:')
# Create a TEXT file for printing the report
report = open(
    'C:\\Users\\alvesd.SMMT\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\python\\16_Geo_Data_Automation\\ONS Data\\Cleansing Report Files\\Geo Cleansing Report.txt',
    'w')
print(
    '########################################################################################################################################################################################################################################################################################################################################')
print('Load the data:')
# Load the data: Index of F:\AIS\Motorparc\Standard Reports\2020 Census\
Regional_Data = pd.read_excel(
    'C:\\Users\\alvesd.SMMT\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\python\\16_Geo_Data_Automation\\ONS Data\\Regional Data1.xlsx')

print(
    '########################################################################################################################################################################################################################################################################################################################################')
print('Load the data:')
# Load the data: Index of F:\AIS\Motorparc\Standard Reports\2021 Census\
Doogal_Data = pd.read_excel(
    'C:\\Users\\alvesd.SMMT\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\python\\16_Geo_Data_Automation\\ONS Data\\Doogal_Data.xlsx',
    sheet_name='Sheet1')

print(
    '########################################################################################################################################################################################################################################################################################################################################')
print('Variance results:')
# Slice the data for Variance checks:
Doogal_Data1 = Doogal_Data.rename(columns={'Postcode': 'AlternativePostcode'})
Doogal_Data1 = Doogal_Data1.rename(columns={'Town/Area': 'PostcodeAreaDescription'})

Regional_Data1 = Regional_Data[["AlternativePostcode", "PostcodeAreaDescription", "Region"]]
Doogal_Data1 = Doogal_Data1[["AlternativePostcode", "PostcodeAreaDescription", "Region"]]

print(
    '########################################################################################################################################################################################################################################################################################################################################')
print('Computing Correlation: dfRegional_Data')
# Build, inspect, and check the values of the new data frame:
CorrelationRegional_Data = pd.DataFrame(Regional_Data, columns=["AlternativePostcode", "PostcodeAreaDescription", "Region"])

# or

print('Computing Correlation: dfDoogal_Data')
# Build, inspect, and check the values of the new data frame:
CorrelationDoogal_Data = pd.DataFrame(Doogal_Data, columns=["AlternativePostcode", "PostcodeAreaDescription", "Region"])

def cramers_v(var1, var2):
    crosstab = np.array(pd.crosstab(var1, var2, rownames=None))
    stat = chi2_contingency(crosstab)[0]
    obs = np.sum(crosstab)
    mini = min(crosstab.shape) - 1
    return stat / (obs * mini)


rows = []

for var1 in CorrelationRegional_Data:
    col = []
    for var2 in CorrelationRegional_Data:
        cramers = cramers_v(CorrelationRegional_Data[var1], CorrelationRegional_Data[var2])
        col.append(round(cramers, 2))
    rows.append(col)

cramers_results = np.array(rows)
df = pd.DataFrame(cramers_results, columns=CorrelationRegional_Data.columns, index=CorrelationRegional_Data.columns)

print(df)

print(
    '########################################################################################################################################################################################################################################################################################################################################')


