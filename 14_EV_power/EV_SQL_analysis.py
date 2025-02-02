import pyodbc
import os
import pandas as pd
import numpy as np
from plyer import notification

# Connect to Database
con = pyodbc.connect(
    "Driver={SQL Server Native Client 11.0};"
    "Server=LAPTOP-1DRQRQOJ\SQL2016;"
    "Database=EV_vehicles;"
    "Trusted_Connection=yes;"
)

#### This code pulls the data from SQL database and save to an Excel file
# SQL command to read the data
sqlQuery = "SELECT TOP(100) * FROM [EV_clean_data]"

# Getting the data from sql into pandas dataframe
df = pd.read_sql(sql=sqlQuery, con=con)

# Export the data to a desktop file
df.to_csv(os.environ["userprofile"] + "\\Desktop\\Python\\14_EV_power\\Data\\" + "EV_UK_Analysis" + ".CSV", index=False)

#Create a TEXT file for our report
report = open('C:\\Users\\Diego Alves\\Desktop\\Python\\14_EV_power\\EV SQL Cleansing Report.txt', 'w')
file = df
report.write('Data shape:' + str(file.shape) + '\n')
report.write('Data type:' + '\n' + str(file.dtypes))
report.write('\n' + str(file.info()))

# Applying data cleansing
## Missing Data Percentage List
for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    percentage = '{} - {}%'.format(col, round(pct_missing*100))
    report.write(percentage + '\n')


## select numeric columns
df_numeric = df.select_dtypes(include=[np.number])
numeric_cols = df_numeric.columns.values
print(numeric_cols)

## select non numeric columns
df_non_numeric = df.select_dtypes(exclude=[np.number])
non_numeric_cols = df_non_numeric.columns.values
print(non_numeric_cols)
report.close()

# Delete rows containing NA values, DFC(data frame cleaned)
dfc = df.dropna()

# Export the data to a desktop file
dfc.to_csv(os.environ["userprofile"] + "\\Desktop\\Python\\14_EV_power\\Data\\" + "EV_UK_power_clean_data" + ".CSV", index=False)

print("Old data frame length:", len(df), "New data frame length:",
      len(df), "Number of rows with at least 1 NA values: ",
      (len(df)-len(dfc)))

# Display notification to user
notification.notify(title="Report Status",
message=f"Cleansing data has been sucessfully saved into Excel. \
          \nTotal Rows: {dfc.shape[0]}\nTotal Columns: {dfc.shape[1]}",
                    timeout=10)

con.close()