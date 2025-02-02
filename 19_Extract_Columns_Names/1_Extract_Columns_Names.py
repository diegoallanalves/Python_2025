import pandas as pd
from IPython.core.display_functions import display
'''
df = pd.read_excel('C:\\Users\\alvesd.SMMT\\Downloads\\1_Final_Geo_Data.xlsx')

print("The DataFrame :")
display(df.head())

# multiple ways of getting column names as list
print("\nThe column headers :")
print("Column headers from list(df.columns.values):",
      list(df.columns.values))
print("Column headers from list(df):", list(df))
print("Column headers from list(df.columns):",
      list(df.columns))
'''
###############################################################################################
df1 = pd.read_csv("C:\\Users\\alvesd.SMMT\\Downloads\\postcodes.csv")

print("The DataFrame :")
display(df1.head())

# multiple ways of getting column names as list
print("\nThe column headers :")
print("Column headers from list(df.columns.values):",
      list(df1.columns.values))
print("Column headers from list(df):", list(df1))
print("Column headers from list(df.columns):",
      list(df1.columns))
##################################################################################################
print(len(df1. columns))

'''
# Slice the data by specific columns:
ONS_Row = ONS_Row[['Postcode', 'In Use?', 'Latitude', 'Longitude', 'Easting', 'Northing', 'Grid Ref', 'County', 'District', 'Ward', 'District Code', 'Ward Code', 'Country', 'County Code', 'Constituency', 'Introduced', 'Terminated', 'Parish', 'National Park', 'Population', 'Households', 'Built up area', 'Built up sub-division', 'Lower layer super output area', 'Rural/urban', 'Region', 'Altitude', 'London zone', 'LSOA Code', 'Local authority', 'MSOA Code', 'Middle layer super output area', 'Parish Code', 'Census output area', 'Constituency Code', 'Index of Multiple Deprivation', 'Quality', 'User Type', 'Last updated', 'Nearest station', 'Distance to station', 'Postcode area', 'Postcode district', 'Police force', 'Water company', 'Plus Code', 'Average Income', 'Sewage Company', 'Travel To Work Area', 'ITL level 2', 'ITL level 3', 'UPRNs', 'Distance to sea', 'LSOA21 Code', 'Lower layer super output area 2021', 'MSOA21 Code', 'Middle layer super output area 2021', 'Census output area 2021']]
'''
