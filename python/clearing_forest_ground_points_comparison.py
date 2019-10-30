import laspy
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# config
filedir = """C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\LiDAR\\19_149\\batch_out\\07_no_buffer\\"""
clearing_in = filedir + """19_149_all_WGS84_utm11_ground-points_upper-clearing.las"""
forest_in = filedir + """19_149_all_WGS84_utm11_ground-points_upper-forest.las"""

# import clearing
datapath = clearing_in
inFile = laspy.file.File(datapath, mode="r")
clearing = pd.DataFrame({'gps_time': [inFile.gps_time], 'scan_angle': [inFile.scan_angle_rank]})
clearing.scan_angle = abs(clearing.scan_angle)

# import forest
datapath = forest_in
inFile = laspy.file.File(datapath, mode="r")
forest = pd.DataFrame({'gps_time': [inFile.gps_time], 'scan_angle': [inFile.scan_angle_rank]})
forest.scan_angle = abs(forest.scan_angle)

# plot
bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
fig, ax = plt.subplots()
ax.hist(clearing.scan_angle, bins, color='blue', alpha=0.5, density=1, label="clearing")
ax.hist(forest.scan_angle, bins, color='red', alpha=0.5, density=1, label="forest")
ax.set_xlabel("Scan angle (deg)")
ax.set_ylabel("Probability density")
ax.set_title(r"Percentage of ground points by scanning angle for clearing and forest")
ax.legend()
fig.tight_layout()
plt.show()
