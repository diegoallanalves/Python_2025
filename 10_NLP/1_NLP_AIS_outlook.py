import win32com.client
import pandas as pd
import os

outlook = win32com.client.Dispatch("Outlook.Application").GetNamespace("MAPI")

inbox = outlook.GetDefaultFolder(6)  # "6" refers to the index of a folder - in this case,
# the inbox. You can change that number to reference
# any other folder
messages = inbox.Items
message = messages("data")
print(message.sender)

folder = outlook.Folders.Item("dalves@smmt.co.uk")
inbox = folder.Folders.Item("Inbox")
messages = inbox.Items

# Initialise Lists
senders = []
addresses = []
subjects = []

os.remove(r'C:\\Users\\alvesd\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\python\\10_NLP\\Addresses.xlsx')

# Iterate through messages for sender name and address
for message in messages:

    try:
        if message.Class == 43:

            if message.SenderEmailType == 'EX':

                if message.Sender.GetExchangeUser() is not None:
                    addresses.append(message.Sender.GetExchangeUser().PrimarySmtpAddress)

                else:
                    addresses.append(message.Sender.GetExchangeDistributionList().PrimarySmtpAddress)

            else:

                addresses.append(message.SenderEmailAddress)

            subjects.append(message.Sender)

    except Exception as e:

        print(e)

# Create Excel file with results
df = pd.DataFrame()
df['Sender'] = senders
df['Address'] = addresses
df['Subject'] = subjects
df.to_excel('C:\\Users\\alvesd\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\python\\10_NLP\\Addresses.xlsx')
