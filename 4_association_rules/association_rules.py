import numpy as np
import matplotlib.pyplot as plt

# importing pandas module
import pandas as pd
from apyori import apriori

store_data = pd.read_csv('C:\\Users\\Diego Alves\\Desktop\\Data_sets\\store_data.csv', header=None)

records = []
for i in range(0, 7501):
    records.append([str(store_data.values[i,j]) for j in range(0, 20)])

association_rules = apriori(records, min_support=0.0056, min_confidence=0.2, min_lift=3, min_length=3)
association_results = list(association_rules)

print(len(association_results))

print(association_results[15])