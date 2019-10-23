import laspy

filedir = """C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\LiDAR\\19_045\\"""
filename_in = """laspy_in.las"""
datapath = filedir + filename_in

# read data in
las_in = laspy.file.File(datapath, mode="r")

# write data out
filename_out = """laspy_out.las"""
datapath = filedir + filename_out
output_file = laspy.file.File(datapath, mode = "w", header = las_in.header)
output_file.points = las_in.points
output_file.close()

# inspect imported 5th return count
sum(las_in.return_num == 5)
