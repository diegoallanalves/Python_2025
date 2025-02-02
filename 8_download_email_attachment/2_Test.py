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