#Reading Outlook emails and downloading attachments

####################################################################################
#Installing the necessary modules
from enum import Enum

class OutlookFolder(Enum):
    olFolderDeletedItems = 3  # The Deleted Items folder
    olFolderOutbox = 4  # The Outbox folder
    olFolderSentMail = 5  # The Sent Mail folder
    olFolderInbox = 6  # The Inbox folder
    olFolderDrafts = 16  # The Drafts folder
    olFolderJunk = 23  # The Junk E-Mail folder

# the module to work with
import win32com.client as win32

# get a reference to Outlook

outlook = win32.Dispatch("Outlook.Application").GetNamespace("MAPI")

# get the Inbox folder (you can a list of all of the possible settings at https://docs.microsoft.com/en-us/office/vba/api/outlook.oldefaultfolders)

inbox = outlook.GetDefaultFolder(OutlookFolder.olFolderInbox.value)

# get subfolder of this
todo = inbox.Folders.Item("Dispute")

# get all the messages in this folder
messages = todo.Items

# check messages exist
if len(messages) == 0:
    print("There aren't any messages in this folder")

# loop over them all
emails = []

for message in messages:
    this_message = (  # get some information about each message in a tuple
        message.Subject,
        message.SenderEmailAddress,
        message.To,
        message.Unread,
        message.Senton.date(),
        message.body,
        message.Attachments
    )
    emails.append(this_message)  # add this tuple of info to a list holding the messages

########################################################################### Show the results:
for email in emails:
    subject, from_address, to_address, if_read, date_sent, body, attachments = email  # unpack the tuple to get at information
    print(subject, to_address)  # show the subject
    if len(attachments) == 0:  # number of attachments
        print("No attachments")
    else:
        for attachment in attachments:
            attachment.SaveAsFile("F:\\AIS\\Diego_F_Drive_Data\\dispute_monitoring\\data\\Output\\" + attachment.FileName)
            print("Saved {0} attachments".format(len(attachments)))
########################################################################### python script to concatenate all the files in the directory into one file:
import pandas as pd
import os
import datetime
import glob
import sys
#################################################################################### Delete the new files to avoid duplications:
os.remove(r'F:\\AIS\\Diego_F_Drive_Data\\dispute_monitoring\\data\\Combined Dispute Monitoring Files.xlsx')
os.remove(r'F:\\AIS\\MVRIS_Procedures\\Disputes\\Dispute Monitoring\\Final Dispute Value.xlsx')
#################################################################################### Delete 30 days old files:
# Delete 30 days old files
retention = 30

current_time = datetime.datetime.now()
retention_time = current_time - datetime.timedelta(days=retention)

log_dir = 'F:\\AIS\\Diego_F_Drive_Data\\dispute_monitoring\\data\\Output\\'
search_log = os.path.join(log_dir, '*.xlsx')

l_logfiles = glob.glob(search_log)

for t_file in l_logfiles:
    t_mod = os.path.getmtime(t_file)
    t_mod = datetime.datetime.fromtimestamp(t_mod)
    print('{0} : {1}'.format(t_file, t_mod))
    if retention_time > t_mod:
        try:
            os.remove(t_file)
            print('Delete: Yes')
        except Exception:
            print('Delete: No')
            print('Error: {0}'.format(sys.exc_info()))
        else:
            print('Delete: Not Required')
#################################################################################### Remove all files with extension:jpg
dir_name = "F:\\AIS\\Diego_F_Drive_Data\\dispute_monitoring\\data\\Output\\"
test = os.listdir(dir_name)

for item in test:
    if item.endswith(".jpg"):
        os.remove(os.path.join(dir_name, item))
#################################################################################### Remove all files with extension:png
dir_name = "F:\\AIS\\Diego_F_Drive_Data\\dispute_monitoring\\data\\Output\\"
test = os.listdir(dir_name)

for item in test:
    if item.endswith(".png"):
        os.remove(os.path.join(dir_name, item))
#################################################################################### Remove all files with extension:gif
dir_name = "F:\\AIS\\Diego_F_Drive_Data\\dispute_monitoring\\data\\Output\\"
test = os.listdir(dir_name)

for item in test:
    if item.endswith(".gif"):
        os.remove(os.path.join(dir_name, item))
####################################################################################
# Load the files
folder = r'F:\\AIS\\Diego_F_Drive_Data\\dispute_monitoring\\data\\Output\\'
df_total = pd.DataFrame()
files = os.listdir(folder)

# Merge the files
for file in files:
    if file.endswith('.xlsx'):
        excel_file = pd.ExcelFile(f'{folder}/{file}')
        sheets = excel_file.sheet_names
        for sheet in sheets:  # Loop through sheets inside an Excel file
            df = excel_file.parse(sheet_name=sheet)
            df_total = df_total.append(df)

# Drop the duplicated
df = df_total.drop_duplicates()

# Add the file into one single file
df.to_excel(f'F:\\AIS\\Diego_F_Drive_Data\\dispute_monitoring\\data\\Combined Dispute Monitoring Files.xlsx')

# Data Preprocessing
df = pd.read_excel('F:\\AIS\\Diego_F_Drive_Data\\dispute_monitoring\\data\\Combined Dispute Monitoring Files.xlsx',
                   sheet_name='Sheet1')
# Drop first column
df = df.iloc[:, 1:]
# Count the unique values and split them
df['Count'] = 1
# Similar names in Column 'Dispute Value'
df.groupby(['Dispute Value']).Count.count().reset_index()
####################################################################################
# Similar names in Column 'Model Group'
grouped = df.groupby('Model Group')
writer = pd.ExcelWriter('F:\\AIS\\MVRIS_Procedures\\Disputes\\Dispute Monitoring\\Final Dispute Value.xlsx',
                        engine='xlsxwriter')
# Split into different tabs and save the file
for name, group in grouped:
    group.to_excel(writer, sheet_name="{}".format(name), index=False)

writer.save()
####################################################################################
