import pandas as pd
import datacompy

# Load the data: Index of F:\AIS\Motorparc\Standard Reports\2020 Census\
df1 = pd.read_excel('C:\\Users\\alvesd\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\data_set\\MP1-2020\\MP1-1F_2019_tested.xlsx', sheet_name = 'Sheet1')
#Drop the columns
df1 = df1.drop(['2006', '2005', 'Pre-2005', 'Total check'], axis=1)
#Load the data: Index of F:\AIS\Motorparc\Standard Reports\2021 Census\
df2 = pd.read_excel('C:\\Users\\alvesd\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\data_set\\MP1-2021\\MP1-1F_tested.xlsx', sheet_name = 'Sheet1')
#Drop the columns
df2 = df2.drop(['2021', '2020', 'Pre-2007', 'Total check'], axis=1)

#Display the report
compare = datacompy.Compare(
df1,
df2,
join_columns=['Unique'], #'MPModelCode', 'MPMakeCode', 'Total'], #You can also specify a list of columns
abs_tol=0.0001,
rel_tol=0,
df1_name='original',
df2_name='new')

print(compare.report())
print(compare.df1_unq_rows)
print(compare.df1_unq_columns())