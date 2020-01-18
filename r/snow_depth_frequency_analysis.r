library(raster)
library(ggplot2)
ras_in <- 'C:/Users/Cob/index/educational/usask/research/masters/data/lidar/analysis/snow_depth/19_045_all_25cm_snow_depth.tif'
sf_in <- 'C:/Users/Cob/index/educational/usask/research/masters/data/lidar/site_library/site_plots.shp'

ds <- raster(x = ras_in)
hist(ds)
plot(ds)

sf <- shapefile(sf_in)
vals <- extract(ds,sf)

for (ii in 1:length(vals)){
  #remove NAs
  vals[[ii]] <- vals[[ii]][!is.na(vals[[ii]])]
  #restructure as df for ggplot
  vals[[ii]] <- data.frame(snow_depth_m = vals[[ii]])
  #assign feature name
  vals[[ii]]$feature <- sf$name[ii]
}

comp <- rbind(vals[[1]], vals[[2]], vals[[3]], vals[[4]], vals[[5]], vals[[6]], vals[[7]])
comp$snow_depth_cm <- comp$snow_depth_m*100

ggplot(data=comp, aes(snow_depth_cm, fill = feature)) +
  geom_density(alpha = 0.2) +
  labs(title ="Snow Depth for 19_045", x = "snow depth (cm)", y = "density") +
  xlim(0, 120)

ggplot(data=comp, aes(snow_depth_cm, fill = feature)) +
  geom_histogram(bins = 30, alpha = 0.2, position = 'identity', aes(y = ..count../sum(..count..))) +
  labs(title ="Snow Depth for 19_045", x = "snow depth (cm)", y = "density") +
  xlim(0, 120)

