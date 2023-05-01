import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np


world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world = world.sort_values('name')
world.loc[world['name'] == 'Dem. Rep. Congo', 'name'] = 'Democratic Republic of Congo'
world.loc[world['name'] == 'Bosnia and Herz.', 'name'] = 'Bosnia and Herzegovina'
world.loc[world['name'] == 'Dominican Rep.', 'name'] = 'Dominican Republic'
world.loc[world['name'] == 'Central African Rep.', 'name'] = 'Central African Republic'
world.loc[world['name'] == 'Eq. Guinea', 'name'] = 'Equatorial Guinea'
world.loc[world['name'] == 'S. Sudan', 'name'] = 'South Sudan'
world.loc[world['name'] == 'Solomon Is.', 'name'] = 'Solomon Islands'
world.loc[world['name'] == 'Timor-Leste', 'name'] = 'Timor'


df_2020 = world.merge(df_2020, left_on='name', right_on='Country')
ax = df_2020.plot(column='CO2 emissions', cmap='OrRd', figsize=(15, 10), scheme='quantiles', legend=True)
ax.set_title('CO2 emissions by country in 2020, million tonnes')