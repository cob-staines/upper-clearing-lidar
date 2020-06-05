library('dplyr')
library('tidyr')
library('ggplot2')

data_res_in <- "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/19_149/19_149_snow_off/OUTPUT_FILES/DEM/19_149_all_200311_628000_5646525dem_all_res_ground_pount_samples.csv"

#import points
data_res = read.csv(data_res_in, header=TRUE, na.strings = c("NA",""), sep=",")
# clean up resolution column names
colnames(data_res)[13:18] = c(".01", ".04", ".10", ".25", ".50", "1")

# filter quality
data_res = data_res[data_res$qc_flag == 0,]

# define standard error
sefx = function(data){
  sd(data, na.rm=TRUE)/sqrt(length(na.omit(data)))
}

# define rmse
rmse = function(difdata){
  sqrt(sum(difdata^2, na.rm = TRUE))/sqrt(length(na.omit(difdata)))
}

data_cover_rmse <- data_res %>% 
  gather(res, DSM_elev, 13:18, factor_key=FALSE) %>%
  mutate(error_elev = WGS84.Ell - DSM_elev) %>%
  group_by(res, cover) %>%
  summarise(rmse_elev = rmse(error_elev), n = length(na.omit(error_elev)))

# res as numeric
data_cover_rmse$res = as.numeric(data_cover_rmse$res)

ggplot(data_cover_rmse, aes(x=res, y=rmse_elev, color=cover)) +
  geom_line() + 
  geom_point() +
  labs(title ='RMSE of LiDAR DEM from GNSS points across resolutions and cover types', x = "Resolution (m)", y = "RMSE (m)", color = "Cover")

ggplot(data_cover_rmse, aes(x=res, y=n, color=cover)) +
  geom_line() + 
  geom_point() +
  labs(title ='Coverage of GNSS points by LiDAR DEM across resolutions and cover types', x = "Resolution (m)", y = "Samples (n)", color = "Cover")

# generalized for all cover types
data_rmse<- data_res %>% 
  gather(res, DSM_elev, 13:18, factor_key=FALSE) %>%
  mutate(error_elev = WGS84.Ell - DSM_elev) %>%
  group_by(res) %>%
  summarise(rmse_elev = rmse(error_elev), n = length(na.omit(error_elev)))

# res as numeric
data_rmse$res = as.numeric(data_rmse$res)

ggplot(data_rmse, aes(x=res, y=rmse_elev)) +
  geom_line() + 
  geom_point() +
  labs(title ='RMSE of LiDAR DEM from GNSS points across resolutions', x = "Resolution (m)", y = "RMSE (m)")

ggplot(data_rmse, aes(x=res, y=n)) +
  geom_line() + 
  geom_point() +
  labs(title ='Coverage of GNSS points by LiDAR DEM across resolutions', x = "Resolution (m)", y = "Total Samples (m)")

# looks good... but is it really? dropping so many points, again, compromising with coverage for accuracy of a scant few points
