from libraries import raslib

ras_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\RAS\\19_149_ground_point_density_r.10m.bil"

ras = raslib.raster_load(ras_in)
ras.data[ras.data == ras.no_data] = 0

ras_out = ras_in.replace(".bil", "_nona.tif")
raslib.raster_save(ras, ras_out)