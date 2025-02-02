import pandas as pd
import json

with open('C:\\Users\\alvesd\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\data_set\\Team.json', encoding='utf-8') as inputfile:
    df = json.loads(inputfile)

'''

df.to_csv('C:\\Users\\alvesd\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\python\\csvfile.csv', encoding='utf-8', index=False)

'''