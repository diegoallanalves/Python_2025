import win32com.client
import sys
import unicodecsv as csv

output_file = open(
    'C:\\Users\\alvesd\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\python\\10_NLP\\outlook_farming_001.csv',
    'wb')
output_writer = csv.writer(output_file, delimiter=";", encoding='latin2')

outlook = win32com.client.Dispatch("Outlook.Application").GetNamespace("MAPI")
inbox = outlook.GetDefaultFolder(6)  # "6" refers to the index of a folder - in this case, # the inbox.

for folder in outlook.Folders:
    print(folder.Name)
