# Reading Outlook emails and downloading attachments
print(
    "######################################################################################################################################################################")

# import mail
from PIL import Image

img = Image.open(
    r"C:\\Users\\alvesd\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\python\\24_Sentiment_Analysis\\Crisp_Analysis_Pic.jpg")
# print(img)

print(
    "######################################################################################################################################################################")

# https://python-forum.io/thread-24810.html
# Trying to export email body in Excel
# https://stackoverflow.com/questions/56445868/trying-to-export-email-body-in-excel
# How to get all the latests emails: https://stackoverflow.com/questions/54668793/python-script-not-getting-most-recent-mail-from-outlook

print(
    "######################################################################################################################################################################")

print('Step 1 - Load the codes libraries:')

import win32com.client
# other libraries to be used in this script
import os

print(
    "######################################################################################################################################################################")

print('Step 2 - Delete the old files from the directory:')
path = r'C:\\Users\\alvesd\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\python\\25_Issue_Log_Email_Extraction'
os.chdir(path)
for file in os.listdir(path):
    if file.endswith('.xlsx') or file.endswith('.xls'):
        # print(file)
        os.remove(file)

for file in os.listdir(path):
    if file.endswith('.xlsx') or file.endswith('.csv'):
        # print(file)
        os.remove(file)

for file in os.listdir(path):
    if file.endswith('.xlsx') or file.endswith('.txt'):
        # print(file)
        os.remove(file)

print(
    "######################################################################################################################################################################")

print("Step 4 - Test Connection Outlook:")

import win32com.client  # pip install pywin32 if not installed

# Connect to Outlook by MAPI
outlook = win32com.client.Dispatch("Outlook.Application").GetNamespace("MAPI")

inbox = outlook.GetDefaultFolder(6)  # "6" refers to the index of a folder - in this case,
# the inbox. You can change that number to reference
# any other folder
messages = inbox.Items
messages.Sort("[ReceivedTime]", True)
message = messages.GetFirst()
body_content = message.body
sender = message.sender
subject = message.Subject
date = message.senton.date()
time = message.senton.time()
attachments = message.Attachments
# print(body_content)

print(
    "######################################################################################################################################################################")

print("Step 5 - Extract Emails to Excel:")

# !/usr/bin/python
"""Script to fetch email from outlook."""
import win32com.client
import pandas as pd
from datetime import datetime, timedelta

def extract(count):
    """Get emails from outlook."""
    items = []
    outlook = win32com.client.Dispatch("Outlook.Application").GetNamespace("MAPI")
    inbox = outlook.GetDefaultFolder(6)  # "6" refers to the inbox
    messages = inbox.Items
    messages.Sort("[ReceivedTime]", True)
    message = messages.GetFirst()
    # message = messages.Sort("[ReceivedTime]", True)
    received_dt = datetime.now() - timedelta(days=3)
    received_dt = received_dt.strftime('%m/%d/%Y %H:%M %p')
    # messages = messages.Sort("[ReceivedTime]", True)
    messages = messages.Restrict("[ReceivedTime] >= '" + received_dt + "'")
    i = 0
    while message:
        try:
            msg = dict()
            msg["Subject"] = getattr(message, "Subject", "<UNKNOWN>")
            msg["SentOn"] = getattr(message, "SentOn", "%m/%d/%Y %H:%M %p")
            # msg["EntryID"] = getattr(message, "EntryID", "<UNKNOWN>")
            msg["Sender"] = getattr(message, "Sender", "smmtdata@smmt.co.uk")
            # msg["Size"] = getattr(message, "Size", "<UNKNOWN>")
            msg["Body"] = getattr(message, "Body", "<UNKNOWN>")
            items.append(msg)
        except Exception as ex:
            print("Error processing mail", ex)
        i += 1
        if i < count:
            message = messages.GetNext()
        else:
            return items

    return items


def show_message(items):
    """Show the messages."""
    items.sort(key=lambda tup: tup["SentOn"])
    for i in items:
        print(i["SentOn"], i["Sender"], i["Subject"], i["Body"])  # i["Body"]


def main():
    items = extract(527)
    df = pd.DataFrame(data=items)
    df["SentOn"] = df["SentOn"].dt.tz_convert(None)
    # show_message(items)
    writer = pd.ExcelWriter(
        r'C:\\Users\\alvesd\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\python\\25_Issue_Log_Email_Extraction\\messages_list.xlsx',
        engine='xlsxwriter',
        engine_kwargs={'options': {'strings_to_urls': False}}
    )
    df.to_excel(writer)
    writer.close()


if __name__ == "__main__":
    main()

print(
    "######################################################################################################################################################################")

print("Step 6 - Load the Data:")

emails = pd.read_excel(
    "C:\\Users\\alvesd\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\python\\25_Issue_Log_Email_Extraction\\messages_list.xlsx")

print(emails.head())

print(
    "######################################################################################################################################################################")

print("Step 7 - Remove Special Characters:")

import openpyxl

emails['Body'] = emails['Body'].astype(str).apply(openpyxl.utils.escape.unescape)

print(emails.head())

import numpy as np

# Remove special characters from file:
emails['Body'] = emails['Body'].replace([r'[<%]', r'^\s*$'], [' ', np.nan], regex=True)

print('Step: Save file')

writer = pd.ExcelWriter(
    r'C:\\Users\\alvesd\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\python\\25_Issue_Log_Email_Extraction\\messages_list.xlsx',
    engine='xlsxwriter',
    engine_kwargs={'options': {'strings_to_urls': False}}
)
emails.to_excel(writer)

writer.close()

emails.to_csv(
    r'C:\\Users\\alvesd\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\python\\25_Issue_Log_Email_Extraction\\messages_list.csv',
    index=False)

print(
    "######################################################################################################################################################################")

import pandas as pd

read_file = pd.read_excel(
    r'C:\\Users\\alvesd\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\python\\25_Issue_Log_Email_Extraction\\messages_list.xlsx',
    sheet_name='Sheet1')
read_file.to_csv(
    r'C:\\Users\\alvesd\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\python\\25_Issue_Log_Email_Extraction\\messages_list.txt',
    index=None, header=True)

print(
    "######################################################################################################################################################################")

print("Step 8 - Drop NA Rows:")

emails.dropna(axis=0, how='any', inplace=True)

# Remove special characters
import re

emails['Body'] = emails['Body'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])

# Drop Null Rows

emails['Body'] = emails['Body'].replace(" ", "")
emails['Body'] = emails['Body'].replace("  ", "")
emails['Body'] = emails['Body'].replace("   ", "")
emails['Body'].replace('', np.nan, inplace=True)
emails.dropna(subset=['Body'], inplace=True)

# Dropping ALL duplicate values
emails.drop_duplicates(subset="Body",
                       keep=False, inplace=True)

print(
    "######################################################################################################################################################################")

print("Step 9 - Sort by Date:")

# Sort by Date:
# Sort the list in ascending order of dates
emails.sort_values(by='SentOn', key=pd.to_datetime, inplace=True)

# save the cleansed file
emails.to_excel(
    r'C:\\Users\\alvesd\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\python\\25_Issue_Log_Email_Extraction\\messages_list.xlsx',
    index=None, header=True)

print(
    "######################################################################################################################################################################")

print("Step 10 - Auto Fit Rows to Column:")

from win32com.client import Dispatch

excel = Dispatch('Excel.Application')
wb = excel.Workbooks.Open(
    "C:\\Users\\alvesd\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\python\\25_Issue_Log_Email_Extraction\\messages_list.xlsx")

# Activate second sheet
excel.Worksheets(1).Activate()

# Autofit column in active sheet
excel.ActiveSheet.Columns.AutoFit()

# Or simply save changes in a current file
wb.Save()

wb.Close()

print(
    "######################################################################################################################################################################")

print("Done up to here")
