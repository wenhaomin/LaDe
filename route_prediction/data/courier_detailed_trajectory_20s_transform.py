import pickle
import pandas as pd
import os

from pyproj import Transformer
transformer = Transformer.from_crs('EPSG:3857', 'EPSG:4326')

def get_workspace():
    """
    get the workspace path
    :return:
    """
    cur_path = os.path.abspath(__file__)
    file = os.path.dirname(cur_path)
    return file

ws =  get_workspace()
traj = pd.read_pickle(ws + '/courier_detailed_trajectory_20s.pkl')
traj_lat_2 = traj['lng'].values + 2379974.967801108
traj_lng_2 = traj['lat'].values + 10143031.442489658
traj_lng = traj_lng_2 - 2379974.967801108
traj_lat = traj_lat_2 - 10143031.442489658
traj['lng'] = traj_lng
traj['lat'] = traj_lat
traj_lng_2 = traj['lng'].values + 2379974.967801108
traj_lat_2 = traj['lat'].values + 10143031.442489658 # lat + mean_lat -> lng
traj_loc = transformer.transform(traj_lng_2, traj_lat_2) # transform(lng, lat)-> lat, lng
"""
traj_loc为真实经纬度
"""