import aspose.words as aw

# Create and save a simple document
doc = aw.Document('C:\\Users\\alvesd\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\python\\4_Cleansing\\MP_Cleansing\\Cleansing_Report.txt')
#doc1 = aw.Document('C:\\Users\\alvesd\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\python\\4_Cleansing\\df1 visualizing_missing_data_with_barplot_Seaborn_distplot.png')
builder = aw.DocumentBuilder(doc)
builder.writeln("Cleansing")
doc.save("C:\\Users\\alvesd\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\python\\4_Cleansing\\MP_Cleansing\\Cleansing_Report.docx")
#doc1.save("C:\\Users\\alvesd\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\python\\4_Cleansing\\DCD_19\\SVT_Cleansing_Report.docx")

#doc = aw.Document('C:\\Users\\alvesd\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\python\\4_Cleansing\\SVT_Cleansing\\SVT_Cleansing_Report.txt')
#doc.save('C:\\Users\\alvesd\\OneDrive - smmt.co.uk\\Desktop\\Diego_work_folder\\python\\4_Cleansing\\SVT_Cleansing\\SVT_Cleansing_Report.html')