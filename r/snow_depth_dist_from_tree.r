library(raster)
library(ggplot2)
ds_in <- 'C:/Users/Cob/index/educational/usask/research/masters/data/lidar/analysis/snow_depth/19_045_all_25cm_snow_depth.tif'
dt_in <- 'C:/Users/Cob/index/educational/usask/research/masters/data/lidar/19_149/19_149_all_test/OUTPUT_FILES/CHM/19_149_all_test_628000_564657pit_free_chm_.25m_parent_dist.tif'

depth_snow <- raster(x = ds_in)
dist_tree <- raster(x = dt_in)
hist(depth_snow)
hist(dist_tree)
