import pyodbc
import pandas as pd
import numpy as np
import datacompy
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import aspose.words as aw

# Load the data: SVT12 as df1 ##########################################################################Beginning of Part1
#Connet to Database
con = pyodbc.connect("driver={SQL Server Native Client 11.0};"
                     "Server=GPS-SRV12;"
                     "database=Parc;"
                     'UID="SMMT\alvesd";'
                     'PWD=;'
                     "Trusted_connection=yes;")

#### This code pulls the data from SQL database and save to an Excel file
# SQL command to read the data

sqlQuery = "SELECT * FROM [Parc].[dbo].[SVT_12];"

# Getting the data from sql into pandas dataframe
df = pd.read_sql(sql=sqlQuery, con=con)
#Create a unique column
#df['Unique'] = df['MVRIS_MODEL_CODE'].astype(str) + '/' + df['MVRISCODE'].astype(str) + '/' + df['MP_RANGE_CODE'].astype(str) + '/' + df["MP_MODEL_CODE"].astype(str) #+ '/' + df["Trans"].astype(str)
df['Unique'] = df['MVRISCODE'].astype(str)

# Move the unique column to the first position
first_column = df.pop('Unique')
df.insert(0, 'Unique', first_column)
# Export Pandas DataFrame to xlsx
df.to_excel('C:\\Users\\alvesd\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\data_set\\SVT_2021\\SVT_12.xlsx',encoding='utf-8', index=False)
df1 = pd.read_excel('C:\\Users\\alvesd\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\data_set\\SVT_2021\\SVT_12.xlsx', sheet_name = 'Sheet1')

# The info method prints to the screen the number of non-missing values of each column
#df1.info()

######################################################################################################End of Part1
# Load the data: SVT as df3 ##########################################################################Beginning of Part2
df2 = pd.read_excel('C:\\Users\\alvesd\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\data_set\\SVT_2021\\SVT.xlsx', sheet_name = 'SVT')
#df2['Unique'] = df2['MVRIS_MODEL_CODE'].astype(str) + '/' + df2['MVRISCODE'].astype(str) + '/' + df2['MP_RANGE_CODE'].astype(str) + '/' + df2['MP_MODEL_CODE'].astype(str)# + '/' + df1["Body"].astype(str) + '/' + df1["Trans"].astype(str)
df2['Unique'] = df2['MVRISCODE'].astype(str)
# Move the unique column to the first position
first_column = df2.pop('Unique')
df2.insert(0, 'Unique', first_column)
# Export Pandas DataFrame to xlsx
df2.to_excel('C:\\Users\\alvesd\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\data_set\\SVT_2021\\SVT_unique.xlsx',encoding='utf-8', index=False)
df3 = pd.read_excel('C:\\Users\\alvesd\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\data_set\\SVT_2021\\SVT_unique.xlsx', sheet_name = 'Sheet1')

# The info method prints to the screen the number of non-missing values of each column
#df3.info()
#################################################################################################
# Replace empty values with Null
df1.replace(r'^\s*$', np.nan, regex=True)
df3.replace(r'^\s*$', np.nan, regex=True)

# Now replace null values in a dataframe using another dataframe
#Column by column]
df3.loc[:, ['DVLA_DESCRIPTION']] = df1[['DVLA_DESCRIPTION']]
df3.loc[:, ['INTRO_DATE']] = df1[['INTRO_DATE']]
df3.loc[:, ['ENGINE_CONFIGURATION']] = df1[['ENGINE_CONFIGURATION']]
df3.loc[:, ['FUEL_DELIVERY']] = df1[['FUEL_DELIVERY']]
df3.loc[:, ['POWER_KW']] = df1[['POWER_KW']]
df3.loc[:, ['BORE']] = df1[['BORE']]
df3.loc[:, ['CALC_BHP']] = df1[['CALC_BHP']]
df3.loc[:, ['POWER_RPM']] = df1[['POWER_RPM']]
#Replace the entire data
df1.update(df3)

# Export Clean Data using Pandas DataFrame to xlsx
df3.to_excel('C:\\Users\\alvesd\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\data_set\\SVT_2021\\SVT_cleaned_file.xlsx', encoding='utf-8')

#######################################################################################################End of Part2

#Create a TEXT file for printing the report
report = open('C:\\Users\\alvesd\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\python\\4_Cleansing\\SVT_Cleansing\\SVT_Cleansing_Report.txt', 'w')

#Comparison Between df1 + df2 ##########################################################################Beginning of Part3

# Display the report. Variance results:
compare = datacompy.Compare(
    df1,
    df2,
    join_columns=['Unique'],  # You can also specify a list of columns
    abs_tol=0.0001,
    rel_tol=0,
    df1_name='original',
    df2_name='new')
# Save the variance results to a TXT file
report.write('\n' + '\n' + 'Cross variance results:' + str(compare.report()))
report.write('\n' + '\n' + 'Unique rows: SVT_12' + str(compare.df1_unq_rows))
report.write('\n' + '\n' + 'Unique rows: SVT' + str(compare.df2_unq_rows))
report.write('\n' + '\n' + 'Unique columns: SVT_12' + '\n' + str(compare.df1_unq_columns()))
report.write('\n' + '\n' + 'Unique columns: SVT' + '\n' + str(compare.df2_unq_columns()))

###########################################################################################################End of Part3

#####################################################################################################Beginning of Part4

# The isna method returns a DataFrame of all boolean values (True/False).
df1_null1 = df1.isna()
df1_null = df1_null1.sum()
report.write('\n' + '\n' + 'Nulls values by columns in the data: [Parc].[dbo].[SVT_12]  ' + str(df1_null.sum) + '\n' + '\n')

## Missing Data Percentage List
for col in df1.columns:
    pct_missing = np.mean(df1[col].isnull())
    percentage = '{} - {}%'.format(col, round(pct_missing * 100))
    report.write('Missing Data by Percentage [Parc].[dbo].[SVT_12]: ' + str(percentage) + '\n')
    # print(percentage)

# Visualizing Missing Data using Seaborn heatmap()
plt.figure(figsize=(10, 6))
sns.heatmap(df1.isna().transpose(),
            cmap="YlGnBu",
            cbar_kws={'label': 'Missing Data'})
plt.savefig("[Parc].[dbo].[SVT_12] visualizing_missing_data_with_heatmap_Seaborn_Pytthon.png", dpi=100)
# plt.show()

# Visualizing Missing Data using Seaborn displot()
plt.figure(figsize=(10, 6))
sns.displot(
    data=df1.isna().melt(value_name="missing").astype(str),
    y="variable",
    hue="missing",
    multiple="fill",
    aspect=1.25
)
plt.savefig("[Parc].[dbo].[SVT_12] visualizing_missing_data_with_barplot_Seaborn_distplot.png", dpi=100)

###########################################################################################################End of Part4

#####################################################################################################Beginning of Part5

# The isna method returns a DataFrame of all boolean values (True/False).
df2_null1 = df2.isna()
df2_null = df2_null1.sum()
report.write('\n' + '\n' + 'Nulls values by columns in the data: SVT ' + str(df2_null.sum) + '\n' + '\n')

## Missing Data Percentage List
for col in df2.columns:
    pct_missing = np.mean(df2[col].isnull())
    percentage = '{} - {}%'.format(col, round(pct_missing * 100))
    report.write('Missing Data by Percentage SVT: ' + str(percentage) + '\n')
    # print(percentage)

# Visualizing Missing Data using Seaborn heatmap()
plt.figure(figsize=(10, 6))
sns.heatmap(df2.isna().transpose(),
            cmap="YlGnBu",
            cbar_kws={'label': 'Missing Data'})
plt.savefig("SVT visualizing_missing_data_with_heatmap_Seaborn_Pytthon.png", dpi=100)
# plt.show()

# Visualizing Missing Data using Seaborn displot()
plt.figure(figsize=(10, 6))
sns.displot(
    data=df2.isna().melt(value_name="missing").astype(str),
    y="variable",
    hue="missing",
    multiple="fill",
    aspect=1.25
)
plt.savefig("SVT visualizing_missing_data_with_barplot_Seaborn_distplot.png", dpi=100)
###########################################################################################################End of Part5

con.close()
sys.exit()





