import folium
import pandas

data = pandas.read_csv('C:\\Users\\Diego Alves\\Desktop\\Data_sets\\EV_clean_data.csv')

# Create toolip

tooltip = 'Payment'

data = data.dropna(subset=['latitude'])
data = data.dropna(subset=['longitude'])

latitude = list(data["latitude"])
longitude = list(data["longitude"])

map = folium.Map(location=[51.509865, -0.118092], zoom_start=4, control_scale=True)

fg = folium.FeatureGroup(name="My Map")

for (index, row) in data.iterrows():
    c1 = fg.add_child(folium.Marker(location=[row.loc['latitude'], row.loc['longitude']],
                                    popup=row.loc['payment'],
                                    tooltip=tooltip,
                                    icon=folium.Icon(color='green')))

child = fg.add_child(folium.Marker(location=[31.5204, 74.5387], popup="EV charger station", icon= folium.Icon(color='green')))

map.add_child(fg)

map.save("payment.html")

