import folium
import pandas

data = pandas.read_csv('C:\\Users\\Diego Alves\\Desktop\\Data_sets\\EV_clean_data.csv')

data = data.dropna(subset=['longitude'])
data = data.dropna(subset=['latitude'])

lat = list(data["latitude"])
lon = list(data["longitude"])

map = folium.Map(location=[51.509865, -0.118092], zoom_start=4, control_scale=True)

fg = folium.FeatureGroup(name="My Map")

for lt, ln in zip(lat, lon):
    c1 = fg.add_child(folium.Marker(location=[lt, ln], popup="EV Charge Station",icon=folium.Icon(color='green')))

child = fg.add_child(folium.Marker(location=[31.5204, 74.5387], popup="EV charger station", icon= folium.Icon(color='green')))

map.add_child(fg)

map.save("EV_charge_stations.html")







