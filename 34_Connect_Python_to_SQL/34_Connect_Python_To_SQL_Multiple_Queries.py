print('Download the Libraries:')
from sqlalchemy.engine import URL
from sqlalchemy.sql import text
from sqlalchemy import create_engine
import sqlalchemy as sa
import numpy as np
import csv
from sqlalchemy.orm import sessionmaker
import os
from io import BytesIO
from collections import OrderedDict
import pandas as pd
import requests, zipfile
from win32com.client import Dispatch

#######################################################################################################

connection_string = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=GPS-SRV20;DATABASE=AnonymisedData;UID=SMMT\\alvesd;PWD=;Trusted_connection=yes"
connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": connection_string})

engine16 = create_engine(connection_url)

Session = sessionmaker(bind=engine16)
session = Session()
session.execute(text('''Drop table IF EXISTS Makefindtemp;'''))
session.execute(text('''Drop table IF EXISTS Makefindtemp2;'''))
session.execute(text('''Drop table IF EXISTS Makefindtemp3;'''))
session.execute(text('''Drop table IF EXISTS Makefindtemp4;'''))
session.execute(text('''Drop table IF EXISTS Makefindtemp5;'''))
session.execute(text('''Drop table IF EXISTS Makefindtemp6;'''))
session.execute(text('''Drop table IF EXISTS Makefindtemp7;'''))
session.execute(text('''Drop table IF EXISTS Makefindtemp8;'''))
session.execute(text('''Drop table IF EXISTS Makefindtemp9;'''))
session.execute(text('''Drop table IF EXISTS DVLAreceivedMake;'''))
session.execute(text('''Drop table IF EXISTS MCRISTemp;'''))
session.execute(text('''Drop table IF EXISTS MVRISRANGE;'''))

session.commit()
session.close()

#######################################################################################################
connection_string = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=GPS-SRV20;DATABASE=AnonymisedData;UID=SMMT\\alvesd;PWD=;Trusted_connection=yes"
connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": connection_string})

engine1 = create_engine(connection_url)

Session = sessionmaker(bind=engine1)
session = Session()
session.execute(text('''select distinct DVLAMakeCode, DVLAModelCode, TRIM(Descriptive) as Descriptive into Makefindtemp from AnonRaw_Staging where DVLAModelCode = '000' and Descriptive <> '                              ' order by Descriptive;'''))
session.commit()
session.close()
#######################################################################################################

#######################################################################################################
connection_string = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=GPS-SRV20;DATABASE=AnonymisedData;UID=SMMT\\alvesd;PWD=;Trusted_connection=yes"
connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": connection_string})

engine2 = create_engine(connection_url)

Session = sessionmaker(bind=engine2)
session = Session()
session.execute(text('''Select distinct DVLAMakeCode, DVLAModelCode, trim(left (Descriptive, charindex(' ',Descriptive,1))) as Make, IDENTITY (int,1,1) AS ID into Makefindtemp2 from Makefindtemp order by Make, DVLAMakecode, DVLAModelCode;'''))

session.commit()
session.close()
#######################################################################################################

#######################################################################################################
connection_string = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=GPS-SRV20;DATABASE=AnonymisedData;UID=SMMT\\alvesd;PWD=;Trusted_connection=yes"
connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": connection_string})

engine3 = create_engine(connection_url)

Session = sessionmaker(bind=engine3)
session = Session()
session.execute(text('''Select distinct Make, IDENTITY (int,1,1) AS ID2 into DVLAreceivedMake from DVLA_Received;'''))

session.commit()
session.close()
#######################################################################################################

#######################################################################################################
connection_string = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=GPS-SRV20;DATABASE=AnonymisedData;UID=SMMT\\alvesd;PWD=;Trusted_connection=yes"
connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": connection_string})

engine4 = create_engine(connection_url)

Session = sessionmaker(bind=engine4)
session = Session()
session.execute(text('''Select Make, ID into Makefindtemp3 from Makefindtemp2 except select DV.Make, ID2 from DVLAreceivedMake as DV;'''))

session.commit()
session.close()
#######################################################################################################

#######################################################################################################
connection_string = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=GPS-SRV20;DATABASE=AnonymisedData;UID=SMMT\\alvesd;PWD=;Trusted_connection=yes"
connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": connection_string})

engine5 = create_engine(connection_url)

Session = sessionmaker(bind=engine5)
session = Session()
session.execute(text('''Select distinct Manufacturer, IDENTITY (int,1,1) AS ID6 into MCRIStemp from MCRISDec2020'''))

session.commit()
session.close()
#######################################################################################################

#######################################################################################################
connection_string = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=GPS-SRV20;DATABASE=AnonymisedData;UID=SMMT\\alvesd;PWD=;Trusted_connection=yes"
connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": connection_string})

engine6 = create_engine(connection_url)

Session = sessionmaker(bind=engine6)
session = Session()
session.execute(text('''Select Make, ID into Makefindtemp4 from Makefindtemp3 except select  Manufacturer, ID6 from MCRISTemp;'''))

session.commit()
session.close()
#######################################################################################################

#######################################################################################################
connection_string = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=GPS-SRV20;DATABASE=AnonymisedData;UID=SMMT\\alvesd;PWD=;Trusted_connection=yes"
connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": connection_string})

engine7 = create_engine(connection_url)

Session = sessionmaker(bind=engine7)
session = Session()
session.execute(text('''Select distinct trim(Marque) as [Marque], trim([Range]) as [Range], IDENTITY (int,1,1) AS ID3 into MVRISRANGE from MVRIS_Codebook;'''))

session.commit()
session.close()
#######################################################################################################

#######################################################################################################
connection_string = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=GPS-SRV20;DATABASE=AnonymisedData;UID=SMMT\\alvesd;PWD=;Trusted_connection=yes"
connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": connection_string})

engine8 = create_engine(connection_url)

Session = sessionmaker(bind=engine8)
session = Session()
session.execute(text('''Select Make, ID into Makefindtemp5 from Makefindtemp4 except select [Range], ID3 from MVRISRANGE;'''))

session.commit()
session.close()
#######################################################################################################

#######################################################################################################
connection_string = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=GPS-SRV20;DATABASE=AnonymisedData;UID=SMMT\\alvesd;PWD=;Trusted_connection=yes"
connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": connection_string})

engine9 = create_engine(connection_url)

Session = sessionmaker(bind=engine9)
session = Session()
session.execute(text('''Select  Make, Marque, [Range] into Makefindtemp6 from Makefindtemp5 cross join MVRISRANGE;'''))

session.commit()
session.close()
#######################################################################################################

#######################################################################################################
connection_string = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=GPS-SRV20;DATABASE=AnonymisedData;UID=SMMT\\alvesd;PWD=;Trusted_connection=yes"
connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": connection_string})

engine10 = create_engine(connection_url)

Session = sessionmaker(bind=engine10)
session = Session()
session.execute(text('''Select distinct Make, charindex (Marque, make) as MakeCheck, charindex ([Range], make) as RangeCheck into Makefindtemp7 from Makefindtemp6 where charindex (Marque, make) + charindex ([Range], make) >0;'''))

session.commit()
session.close()
#######################################################################################################
connection_string = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=GPS-SRV20;DATABASE=AnonymisedData;UID=SMMT\\alvesd;PWD=;Trusted_connection=yes"
connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": connection_string})

engine11 = create_engine(connection_url)

Session = sessionmaker(bind=engine11)
session = Session()
session.execute(text('''Select distinct Make into Makefindtemp8 from Makefindtemp6 except select distinct Make from Makefindtemp7 order by Make;'''))

session.commit()
session.close()
#######################################################################################################

#######################################################################################################
connection_string = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=GPS-SRV20;DATABASE=AnonymisedData;UID=SMMT\\alvesd;PWD=;Trusted_connection=yes"
connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": connection_string})

engine12 = create_engine(connection_url)

Session = sessionmaker(bind=engine12)
session = Session()
session.execute(text('''Select trim(left (Descriptive, charindex(' ',Descriptive,1))) as Make, right ([DateFirstReg],4) as [First reg date],  right([AcquisitionDateCurrentKeeper],4) as [Year], count (*) as [Units] into MakefindTemp9 from [dbo].[AnonRaw_Staging] as ARS where trim(left (Descriptive, charindex(' ',Descriptive,1))) in (Select * from Makefindtemp8) and [NumberofPreviousKeepers] = 0 Group by [Descriptive], [DateFirstReg], [AcquisitionDateCurrentKeeper] Order By [Descriptive], [DateFirstReg], [AcquisitionDateCurrentKeeper];'''))

session.commit()
session.close()
#######################################################################################################

#######################################################################################################
print("Save to Excel")
connection_string = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=GPS-SRV20;DATABASE=AnonymisedData;UID=SMMT\\alvesd;PWD=;Trusted_connection=yes"
connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": connection_string})

engine13 = create_engine(connection_url)

Session = sessionmaker(bind=engine13)
session = Session()

Query = pd.read_sql_query(('''Select [Make], [Year], sum ([Units]) as [New Reg] from MakefindTemp9 where [Make] <> '' and [Year] in ('2022', '2023') GRoup by [Make], [Year];'''),engine13)

DF = pd.DataFrame(Query)

print(DF)

DF.to_excel('C:\\Users\\alvesd\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\python\\28_Used_Automation_SQL\\Search_OEM.xlsx',index=False)

session.commit()
session.close()
#######################################################################################################

#######################################################################################################

connection_string = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=GPS-SRV20;DATABASE=AnonymisedData;UID=SMMT\\alvesd;PWD=;Trusted_connection=yes"
connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": connection_string})

engine17 = create_engine(connection_url)

Session = sessionmaker(bind=engine17)
session = Session()
session.execute(text('''Drop table IF EXISTS Makefindtemp;'''))
session.execute(text('''Drop table IF EXISTS Makefindtemp2;'''))
session.execute(text('''Drop table IF EXISTS Makefindtemp3;'''))
session.execute(text('''Drop table IF EXISTS Makefindtemp4;'''))
session.execute(text('''Drop table IF EXISTS Makefindtemp5;'''))
session.execute(text('''Drop table IF EXISTS Makefindtemp6;'''))
session.execute(text('''Drop table IF EXISTS Makefindtemp7;'''))
session.execute(text('''Drop table IF EXISTS Makefindtemp8;'''))
session.execute(text('''Drop table IF EXISTS Makefindtemp9;'''))
session.execute(text('''Drop table IF EXISTS DVLAreceivedMake;'''))
session.execute(text('''Drop table IF EXISTS MCRISTemp;'''))
session.execute(text('''Drop table IF EXISTS MVRISRANGE;'''))

session.commit()
session.close()
