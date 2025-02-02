# Import libraries
import glob
import pandas as pd

###############################################################################Start of Part1
# The codes below turn all files to one:
# Get CSV files list from a folder
path = 'F:\AIS\Diego_F_Drive_Data\data_set\Test'
csv_files = glob.glob(path + "/*.csv")

# Read each CSV file into DataFrame
# This creates a list of dataframes
df_list = (pd.read_csv(file) for file in csv_files)

# Concatenate all DataFrames: link things together
big_df = pd.concat(df_list, ignore_index=True)

print(big_df)
###############################################################################Part2