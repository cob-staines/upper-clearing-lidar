import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import pandas as pd

traj_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_045\\19_045_all_trajectories_WGS84_utm11N.txt'
traj_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_050\\19_050_all_trajectories_WGS84_utm11N.txt'
traj_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_052\\19_052_all_trajectories_WGS84_utm11N.txt'
traj_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_107\\19_107_all_trajectories_WGS84_utm11N.txt'
traj_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_123\\19_123_all_trajectories_WGS84_utm11N.txt'
traj_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_all_traj.txt'

traj = pd.read_csv(traj_in)
traj.columns = ['gpstime_s', 'roll_deg', 'pitch_deg', 'yaw_deg', 'x_m', 'y_m', 'z_m']
plt.plot(traj.gpstime_s, traj.yaw_deg)