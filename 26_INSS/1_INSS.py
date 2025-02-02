# file:///C:/Users/diego/Downloads/Retenc%CC%A7a%CC%83o%20de%20Tributos%20O%CC%81rga%CC%83os%20Pu%CC%81blicos_Apostila%20atualizada%20IN%202.110_2022.pdf
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

pdf = SimpleDocTemplate("C:\\Users\\diego\\Desktop\\python\\26_INSS\\Retenção de Tributos Órgãos Públicos_Apostila atualizada IN 2.110_2022.pdf")

flow_obj = []

styles = getSampleStyleSheet()

text = ''' mais informacoes clique <a href=http://normas.receita.fazenda.gov.br/sijut2consulta/link.action?idAto=37200 color="BLUE">aqui.</a>
'''

para_text = Paragraph(text, style=styles["Normal"])

flow_obj.append(para_text)

pdf.build(flow_obj)














