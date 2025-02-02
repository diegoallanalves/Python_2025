import pandas as pd
import numpy as np
import networkx as nx
import plotly.express as px
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import sys
from mlxtend.frequent_patterns import association_rules, apriori

warnings.filterwarnings('ignore')

plt.style.use('seaborn')

#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

#Data Preprocessing
df = pd.read_excel('C:\\Users\\alvesd\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\python\\5_Data_Science\\1_Apriori_Data_Shop_2019\\Test Salesforce Reports_.xlsx', sheet_name = 'Data_Shop')
data = df.head(50)

#print(data)

data['ID'] = data.groupby('Transaction_ID', sort=False).ngroup()

first_column = data.pop('ID')
data.insert(0, 'ID', first_column)

#data.info()

print(data.head())

#Let's try and explore which are the top 30 selling products in the bakery using a network diagram.
data_vis = data.copy()
df_network_first = data_vis.groupby("Item_1").sum().sort_values("ID", ascending=False).reset_index()
df_network_first["Type"] = "Datashop"
df_network_first = df_network_first.truncate(before=-1, after=30) # top 30
plt.rcParams['figure.figsize']=(15,15)
j = 0
for i, _ in reversed(list(enumerate(df_network_first['ID']))):
    df_network_first['ID'][j] = i
    j+=1
first_choice = nx.from_pandas_edgelist(df_network_first, source='Type', target="Item_1", edge_attr='ID')
prior = [i['ID'] for i in dict(first_choice.edges).values()]
pos = nx.spring_layout(first_choice)
nx.draw_networkx_nodes(first_choice, pos, node_size=6000, node_color="maroon")
nx.draw_networkx_edges(first_choice, pos, width=prior, alpha=0.4, edge_color='black')
nx.draw_networkx_labels(first_choice, pos, font_size=8, font_family='Franklin Gothic Medium', font_color = 'black')
plt.axis('off')
plt.grid()
plt.title('Top 30 Columns', fontsize=25)
#plt.show()

#Data Modelling
def encoder(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

apriori_data = data.groupby(['ID','Item_1'])['Item_1'].count().reset_index(name ='Count')
apriori_basket = apriori_data.pivot_table(index = 'ID', columns = 'Item_1', values = 'Count', aggfunc = 'sum').fillna(0)
apriori_basket_set = apriori_basket.applymap(encoder)
print(apriori_basket_set.head())

#Let's first analyze the rules with min_support 5% and then for 1% respectively. Both using the metric lift.

# Apriori 1
f_items = apriori(apriori_basket_set, min_support = 0.05, use_colnames = True)
print(f_items.head())

apriori_rules = association_rules(f_items, metric = 'lift', min_threshold = 0.05)
apriori_rules.sort_values('confidence', ascending = False, inplace = True)
print(apriori_rules.head())

# Apriori 2
f_items = apriori(apriori_basket_set, min_support = 0.01, use_colnames = True)
print(f_items.head())

apriori_rules = association_rules(f_items, metric = 'lift', min_threshold = 0.01)
apriori_rules.sort_values('confidence', ascending = False, inplace = True)
print(apriori_rules.head())

#Data Visualizations for Association Rules

apriori_rules['lhs_items'] = apriori_rules['antecedents'].apply(lambda x:len(x) )
apriori_rules[apriori_rules['lhs_items']>1].sort_values('lift', ascending=False).head()
apriori_rules['antecedents_'] = apriori_rules['antecedents'].apply(lambda a: ','.join(list(a)))
apriori_rules['consequents_'] = apriori_rules['consequents'].apply(lambda a: ','.join(list(a)))
pivot = apriori_rules[apriori_rules['lhs_items']>1].pivot(index = 'antecedents_', columns = 'consequents_', values= 'lift')
sns.heatmap(pivot, annot = True)
plt.yticks(rotation=0)
plt.xticks(rotation=90)
#plt.show()

sys.exit()














