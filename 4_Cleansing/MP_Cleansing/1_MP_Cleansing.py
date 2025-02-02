import os
import pandas as pd
import numpy as np
import datacompy
import seaborn as sns
import matplotlib.pyplot as plt
import sys

#Create a TEXT file for printing the report
report = open('C:\\Users\\alvesd\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\python\\4_Cleansing\\MP_Cleansing\\Cleansing_Report.txt', 'w')

###############################################################################Start of Part1

# Load the data: Index of F:\AIS\Motorparc\Standard Reports\2020 Census\
df1 = pd.read_excel('C:\\Users\\alvesd\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\data_set\\MP1-2020\\MP1-1_2020_tested.xlsx', sheet_name = 'Sheet2')

#Join multiple columns and create unique column
df1['Unique'] = df1['MPMakeCode'].astype(str) + '/' + df1['MPRangeCode'].astype(str) + '/' + df1['MPModelCode'].astype(str) + '/' + df1["Body"].astype(str) #+ '/' + df1["Trans"].astype(str)# Join multiple columns and create unique column

# The info method prints to the screen the number of non-missing values of each column
#df1.info()

# The isna method returns a DataFrame of all boolean values (True/False).
df1_null1 = df1.isna()
df1_null = df1_null1.sum()
report.write('\n' + '\n' + 'Nulls values by columns in the data: df1 - 2020 ' + str(df1_null.sum) + '\n' + '\n')

## Missing Data Percentage List
for col in df1.columns:
    pct_missing = np.mean(df1[col].isnull())
    percentage = '{} - {}%'.format(col, round(pct_missing * 100))
    report.write('Missing Data by Percentage df1 - 2020: ' + str(percentage) + '\n')
    # print(percentage)

# Visualizing Missing Data using Seaborn heatmap()
plt.figure(figsize=(10, 6))
sns.heatmap(df1.isna().transpose(),
            cmap="YlGnBu",
            cbar_kws={'label': 'Missing Data'})
plt.savefig("df1 visualizing_missing_data_with_heatmap_Seaborn_Pytthon.png", dpi=100)
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
plt.savefig("df1 visualizing_missing_data_with_barplot_Seaborn_distplot.png", dpi=100)
######################################################################################End of Part1

################################################################################Beginning of Part2

# Load the data: Index of F:\AIS\Motorparc\Standard Reports\2021 Census\
df2 = pd.read_excel('C:\\Users\\alvesd\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\data_set\\MP1-2021\\MP1-1_tested.xlsx', sheet_name = 'Sheet1')

# Join multiple columns and create unique column
df2['Unique'] = df2['MPMakeCode'].astype(str) + '/' + df2['MPRangeCode'].astype(str) + '/' + df2['MPModelCode'].astype(str) + '/' + df2["Body"].astype(str) #+ '/' + df2["Trans"].astype(str)

# The info method prints to the screen the number of non-missing values of each column
#df2.info()

# The ISNA method returns a DataFrame of all boolean values (True/False).
df2_null1 = df2.isna()
df2_null = df2_null1.sum()
# Save the ISNA results to a TXT file
report.write('\n' + '\n' + 'Nulls values by columns in the data df2 - 2021: ' + str(df2_null.sum) + '\n' + '\n')

## Looping: Missing Data Percentage List and save to a TXT file
for col in df2.columns:
    pct_missing = np.mean(df2[col].isnull())
    percentage = '{} - {}%'.format(col, round(pct_missing * 100))
    report.write('Missing Data by Percentage df2 - 2021: ' + str(percentage) + '\n')
    # print(percentage)

# Visualizing Missing Data using Seaborn heatmap()
plt.figure(figsize=(10, 6))
sns.heatmap(df2.isna().transpose(),
            cmap="YlGnBu",
            cbar_kws={'label': 'Missing Data'})
plt.savefig("df2 visualizing_missing_data_with_heatmap_Seaborn_Pytthon.png", dpi=100)
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
plt.savefig("df2 visualizing_missing_data_with_barplot_Seaborn_distplot.png", dpi=100)

#################################################################################End of Part2

###########################################################################Beginning of Part3

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
report.write('\n' + '\n' + 'Unique rows: df1' + str(compare.df1_unq_rows))
report.write('\n' + '\n' + 'Unique rows: df2' + str(compare.df2_unq_rows))
report.write('\n' + '\n' + 'Unique columns: df1' + '\n' + str(compare.df1_unq_columns()))
report.write('\n' + '\n' + 'Unique columns: df2' + '\n' + str(compare.df2_unq_columns()))

######################################################################################End of Part3

# d###########################################################################Beginning of Part4
#Delete old excel file
os.remove("C:\\Users\\alvesd\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\data_set\\MP1-2021\\clean_data\\MP1-1_2020_2021\\MP1-1_2020_verified.xlsx")

# Load the data: Index of F:\AIS\Motorparc\Standard Reports\2020 Census\
d = pd.read_excel('C:\\Users\\alvesd\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\data_set\MP1-2020\\MP1-1_2020_tested.xlsx', sheet_name = 'Sheet2')

#Join multiple columns and create unique column
#df1['Unique'] = df1['MPMakeCode'].astype(str) + '/' + df1['MPRangeCode'].astype(str) + '/' + df1['MPModelCode'].astype(str) + '/' + df1["Make"].astype(str) + '/' + df1["Model Variant"].astype(str) + '/' + df1["Trans"].astype(str) + '/' + df1["CC"].astype(str)
d.replace(r'^\s*$', np.nan, regex=True)
#df1['Unique'] = df1.groupby(['MPMakeCode', 'Model Variant'], sort=False).ngroup() + 1
#df1['Unique'] = pd.factorize(df1['MPMakeCode'].astype(str)+df1['MPRangeCode'].astype(str)+df1['Model Variant'].astype(str)+df1['Trans'].astype(str))[0]
d['Unique'] = d["Model Variant"].astype(str) + "/" + d["Trans"].astype(str)


# Move the unique column to the first position
first_column = d.pop('Unique')
d.insert(0, 'Unique', first_column)

# Export new excel file
d.to_excel('C:\\Users\\alvesd\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\data_set\\MP1-2021\\clean_data\\MP1-1_2020_2021\\MP1-1_2020_verified.xlsx',encoding='utf-8', index=False)
# d#################################################################################End of Part4

# df###########################################################################Beginning of Part5
#Delete old excel file
os.remove("C:\\Users\\alvesd\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\data_set\\MP1-2021\\clean_data\\MP1-1_2020_2021\\MP1-1_2021_verified.xlsx")

# Load the data: Index of F:\AIS\Motorparc\Standard Reports\2021 Census\
df = pd.read_excel('C:\\Users\\alvesd\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\data_set\\MP1-2021\\MP1-1_tested.xlsx', sheet_name = 'Sheet1')

# Join multiple columns and create unique column
#df2['Unique'] = df2['MPMakeCode'].astype(str) + '/' + df2['MPRangeCode'].astype(str) + '/' + df2['MPModelCode'].astype(str) + '/' + df2["Make"].astype(str) + '/' + df2["Model Variant"].astype(str) + '/' + df2["Trans"].astype(str) + '/' + df2["CC"].astype(str)
df.replace(r'^\s*$', np.nan, regex=True)
#df2['Unique'] = df2.groupby(['MPMakeCode', 'Model Variant'], sort=False).ngroup() + 1
#df2['Unique'] = pd.factorize(df2['MPMakeCode'].astype(str)+df2['MPRangeCode'].astype(str)+df2['Model Variant'].astype(str)+df2['Trans'].astype(str))[0]
df['Unique'] = df["Model Variant"].astype(str) + "/" + df["Trans"].astype(str)

# Move the unique column to the first position
first_column = df.pop('Unique')
df.insert(0, 'Unique', first_column)

# Export Pandas DataFrame to xlsx
df.to_excel('C:\\Users\\alvesd\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\data_set\\MP1-2021\\clean_data\\MP1-1_2020_2021\\MP1-1_2021_verified.xlsx',encoding='utf-8', index=False)

# Now replace null values in a dataframe using another dataframe
#d.update(df)
d.fillna(df)

#Delete old excel file
os.remove("C:\\Users\\alvesd\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\data_set\\MP1-2021\\clean_data\\MP1-1_2020_2021\\MP1-1_cleaned.xlsx")

# Export new excel file
df.to_excel('C:\\Users\\alvesd\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\data_set\\MP1-2021\\clean_data\\MP1-1_2020_2021\\MP1-1_cleaned.xlsx', encoding='utf-8')

##################################################################################End of Part5

sys.exit()
