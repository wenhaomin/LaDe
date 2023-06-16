import pickle as pkl
import numpy as np
import pandas as pd
from geopy.distance import geodesic
import os

###### the dataset ########
raw_data_path = './data/pickup_sh.csv'
city_name = 'Shanghai'
abb = 'SH'
saved_dir = './data/sensor_graph/'
file_name = 'adj_mx_delivery_{}.pkl'.format(abb.lower())

#### parameters for weighted adj ####
if abb == 'SH':
    normalized_k =  0.3 #threshold
    s = 5 # scale
elif abb == 'HZ':
    normalized_k =  0.3 #threshold
    s = 5 # scale
elif abb == 'CQ':
    normalized_k =  0.4 #threshold
    s = 10 # scale
#####################################

# read data
df = pd.read_csv(raw_data_path)
# chose city
df_city = df[df.city == city_name]
# a list contain dipan's id
dipan_ids = list(df_city.region_id.unique())

locations = []
for dipan in dipan_ids:
    lng = np.mean(df_city[df_city.region_id == dipan].lng)
    lat = np.mean(df_city[df_city.region_id == dipan].lat)
    locations.append([lat,lng])
    
n_sensors = len(dipan_ids)
dist = np.full((n_sensors, n_sensors), np.inf)
for i in range(n_sensors):
    for j in range(n_sensors):
        dist[i][j] = geodesic(locations[i], locations[j]).kilometers

# Calculates the standard deviation as theta.
distances = dist[~np.isinf(dist)].flatten()
std = distances.std() / s

print(distances.mean(), std, distances.max(), distances.min())

adj_mx = np.exp(-np.square(dist / std))

adj_mx[adj_mx < normalized_k] = 0 # make it sparse

if not os.path.exists(saved_dir):os.mkdir(saved_dir)
with open(os.path.join(saved_dir , file_name),'wb') as f:
    pkl.dump([None,None,adj_mx],f)
print('ADJ saved in {}'.format(os.path.join(saved_dir , file_name)))