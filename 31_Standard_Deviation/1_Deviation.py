# The mean is the sum of all the entries divided by the number of entries. For example, if we have a list of 5 numbers [1,2,3,4,5], then the mean will be (1+2+3+4+5)/5 = 3.
# Standard deviation is a measure of the amount of variation or dispersion of a set of values. We first need to calculate the mean of the values, then calculate the variance, and finally the standard deviation.
# Import statistics Library
import numpy as np
import pandas as pd

my_data = pd.read_excel('C:\\Users\\diego\\Desktop\\python\\31_Standard_Deviation\\deviation.xlsx')

print("The dataframe is :")
my_df = pd.DataFrame(my_data)
print("The standard deviation of column 'Age' is :")
print(my_df['age'].std())

print(
    "The mean is the sum of all the entries divided by the number of entries. For example, if we have a list of 5 numbers [1,2,3,4,5], then the mean will be (1+2+3+4+5)/5 = 3:")
my_df['Average'] = my_df[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12',
                          'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20', 'x21', 'x22', 'x23', 'x24',
                          'x25', 'x26', 'x27', 'x28', 'x29', 'x30', 'x31', 'x32', 'x33', 'x34']].mean(axis=1)
print(my_df['Average'])

print(
    "Variance is the sum of squares of differences between all numbers and means. The Variance deviation of the mean:")
print(np.var(my_df[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12',
                    'x25', 'x26', 'x27', 'x28', 'x29', 'x30', 'x31', 'x32', 'x33', 'x34', 'Average']]))
#print(np.var(my_df['Average']))

print(
    "Standard deviation is a measure of the amount of variation or dispersion of a set of values. We first need to calculate the mean of the values, then calculate the variance, and finally the standard deviation.")
print(np.std(my_df))

