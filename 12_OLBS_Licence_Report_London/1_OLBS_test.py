########################################################################################################################################################################
import pandas as pd

xls = pd.ExcelFile(
    'C:\\Users\\alvesd.SMMT\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\Traffic commissioner\\Test_OLBSLicenceReport_London and the South East of England (1).xlsx')

df = pd.read_excel(xls, 'OLBS_raw')

########################################################################################################################################################################

# Extract Last n characters from right of the column in pandas:
df['Post_Codes'] = df['CorrespondenceAddress'].str[-7:]

# As dataset having lot of extra spaces in cell so lets remove them using strip() function
df['Post_Codes'] = df['Post_Codes'].str.replace(' ', '')

df.to_csv (r'C:\\Users\\alvesd.SMMT\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\Traffic commissioner\\OLBSLicenceReport_London and the South East of England.csv', index = None, header=True)

#df.to_excel("C:\\Users\\alvesd.SMMT\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\Traffic commissioner\\OLBSLicenceReport_London and the South East of England.xlsx")

print(df)

########################################################################################################################################################################

import pandas as pd

coordinates = pd.read_csv(
    'C:\\Users\\alvesd.SMMT\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\Traffic commissioner\\ukpostcodes.csv',
    low_memory=False)

# As dataset having lot of extra spaces in cell so lets remove them using strip() function
coordinates['postcode'] = coordinates['postcode'].str.replace(' ', '')

print(coordinates)

OLBS = pd.read_csv(
    'C:\\Users\\alvesd.SMMT\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\Traffic commissioner\\OLBSLicenceReport_London and the South East of England.csv',
    low_memory=False)

########################################################################################################################################################################
df3 = OLBS.merge(coordinates, left_on='Post_Codes', right_on='postcode')[['Post_Codes', 'GeographicRegion', 'VehiclesSpecified', 'latitude', 'longitude']]

# df3.to_csv (r'C:\\Users\\alvesd.SMMT\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\Traffic commissioner\\Clean.csv', index = None, header=True)

df3.to_excel("C:\\Users\\alvesd.SMMT\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\Traffic commissioner\\Clean.xlsx", index=False)

print(df3)

