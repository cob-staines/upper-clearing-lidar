import laspy
import pandas as pd
import numpy as np


filedir = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_all_test\\OUTPUT_FILES\\LAS\\"
las_in = "19_149_all_test_628000_5646575_vegetation.las"

# import las_in
inFile = laspy.file.File(filedir + las_in, mode="r")
# select dimensions
las_data = pd.DataFrame({'gps_time': inFile.gps_time,
                         'x': inFile.x,
                         'y': inFile.y,
                         'z': inFile.z,
                         'intensity': inFile.intensity})
inFile.close()

# define new origin
origin = np.array([easting_m, northing_m, elevation_m])

p1 = np.array([las_data.x, las_data.y, las_data.z])
p2 = origin - p1

#calculate polar coords
r = np.sqrt(np.sum(p2 ** 2, axis=0))
# drop all beyond 100m
## enter code here##

phi = np.arcsin(z/sqrt(x^2 + y^2))
theda = np.arctan(y/x)
