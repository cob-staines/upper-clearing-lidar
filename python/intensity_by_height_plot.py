import laspy
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time

# config
filedir = """C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\LiDAR\\19_149\\batch_out\\"""
las_in = """forest_ladder_track.las"""

# import las
inFile = laspy.file.File(filedir + las_in, mode="r")
# pull only gps_time
peace = pd.DataFrame(inFile.intensity, inFile.extra_bytes)