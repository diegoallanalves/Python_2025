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

#Get the list of columns as codes: ['unique', 'mpvc', 'mpmakecode', 'mprangecode', 'mpmodelcode', 'make', 'model range', 'model variant', 'series', 'cofo', 'body', 'trans', 'mppt_cc', 'fuel', 'total', '2019', '2018', '2017', '2016', '2015', '2014', '2013', '2012', '2011', '2010', '2009', '2008', '2007']
print(df1.columns.values.tolist())

#Remove words or number and keep the special characters only
#special2 = df1['MPMakeCode'].astype(str).str.replace(r'DVLAPostcode', "")
#special2 = df1['MPMakeCode'].astype(str).str.replace(r'\w', "")
special2 = df1[df1.MPMakeCode.astype(str).str.contains("[^ a-zA-Z0-9]")]
print(special2)
#special2 = df2['MPMakeCode'].astype(str).str.replace(r'\w', "")
#print(special2)
