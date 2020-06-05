library('dplyr')
library('tidyr')
library('ggplot2')

# load data ####
# res_.04
data_res_.04_step_.5_in <- "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/19_045/19_045_snow_on/OUTPUT_FILES/DEM/offset_opt/19_149_all_200311_628000_5646525dem_.04_step_.5_ground_point_samples.csv"
data_res_.04_step_1_in <- "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/19_045/19_045_snow_on/OUTPUT_FILES/DEM/offset_opt/19_149_all_200311_628000_5646525dem_.04_step_1_ground_point_samples.csv"
data_res_.04_step_2_in <- "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/19_045/19_045_snow_on/OUTPUT_FILES/DEM/offset_opt/19_149_all_200311_628000_5646525dem_.04_step_2_ground_point_samples.csv"

#import all points
# step .5
data_res_.04_step_.5 = read.csv(data_res_.04_step_.5_in, header=TRUE, na.strings = c("NA",""), sep=",")
colnames(data_res_.04_step_.5)[18:21] = c(".01", ".03", ".04", ".05")
data_res_.04_step_.5$step = .5
data_res_.04_step_.5$res = .04

# step 1
data_res_.04_step_1 = read.csv(data_res_.04_step_1_in, header=TRUE, na.strings = c("NA",""), sep=",")
colnames(data_res_.04_step_1)[18:21] = c(".01", ".03", ".04", ".05")
data_res_.04_step_1$step = 1
data_res_.04_step_1$res = .04

# step 2
data_res_.04_step_2 = read.csv(data_res_.04_step_2_in, header=TRUE, na.strings = c("NA",""), sep=",")
colnames(data_res_.04_step_2)[18:21] = c(".01", ".03", ".04", ".05")
data_res_.04_step_2$step = 2
data_res_.04_step_2$res = .04

# res_.10
data_res_.10_step_.5_in <- "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/19_045/19_045_snow_on/OUTPUT_FILES/DEM/offset_opt/19_149_all_200311_628000_5646525dem_.10_step_.5_ground_point_samples.csv"
data_res_.10_step_1_in <- "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/19_045/19_045_snow_on/OUTPUT_FILES/DEM/offset_opt/19_149_all_200311_628000_5646525dem_.10_step_1_ground_point_samples.csv"
data_res_.10_step_2_in <- "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/19_045/19_045_snow_on/OUTPUT_FILES/DEM/offset_opt/19_149_all_200311_628000_5646525dem_.10_step_2_ground_point_samples.csv"

#import all points
# step .5
data_res_.10_step_.5 = read.csv(data_res_.10_step_.5_in, header=TRUE, na.strings = c("NA",""), sep=",")
colnames(data_res_.10_step_.5)[18:21] = c(".01", ".03", ".04", ".05")
data_res_.10_step_.5$step = .5
data_res_.10_step_.5$res = .10

# step 1
data_res_.10_step_1 = read.csv(data_res_.10_step_1_in, header=TRUE, na.strings = c("NA",""), sep=",")
colnames(data_res_.10_step_1)[18:21] = c(".01", ".03", ".04", ".05")
data_res_.10_step_1$step = 1
data_res_.10_step_1$res = .10

# step 2
data_res_.10_step_2 = read.csv(data_res_.10_step_2_in, header=TRUE, na.strings = c("NA",""), sep=",")
colnames(data_res_.10_step_2)[18:21] = c(".01", ".03", ".04", ".05")
data_res_.10_step_2$step = 2
data_res_.10_step_2$res = .10

# res_.25
data_res_.25_step_.5_in <- "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/19_045/19_045_snow_on/OUTPUT_FILES/DEM/offset_opt/19_149_all_200311_628000_5646525dem_.25_step_.5_ground_point_samples.csv"
data_res_.25_step_1_in <- "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/19_045/19_045_snow_on/OUTPUT_FILES/DEM/offset_opt/19_149_all_200311_628000_5646525dem_.25_step_1_ground_point_samples.csv"
data_res_.25_step_2_in <- "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/19_045/19_045_snow_on/OUTPUT_FILES/DEM/offset_opt/19_149_all_200311_628000_5646525dem_.25_step_2_ground_point_samples.csv"

#import all points
# step .5
data_res_.25_step_.5 = read.csv(data_res_.25_step_.5_in, header=TRUE, na.strings = c("NA",""), sep=",")
colnames(data_res_.25_step_.5)[18:21] = c(".01", ".03", ".04", ".05")
data_res_.25_step_.5$step = .5
data_res_.25_step_.5$res = .25

# step 1
data_res_.25_step_1 = read.csv(data_res_.25_step_1_in, header=TRUE, na.strings = c("NA",""), sep=",")
colnames(data_res_.25_step_1)[18:21] = c(".01", ".03", ".04", ".05")
data_res_.25_step_1$step = 1
data_res_.25_step_1$res = .25

# step 2
data_res_.25_step_2 = read.csv(data_res_.25_step_2_in, header=TRUE, na.strings = c("NA",""), sep=",")
colnames(data_res_.25_step_2)[18:21] = c(".01", ".03", ".04", ".05")
data_res_.25_step_2$step = 2
data_res_.25_step_2$res = .25
# compile data ####


data = rbind(data_res_.04_step_.5, data_res_.04_step_1, data_res_.04_step_2, data_res_.10_step_.5, data_res_.10_step_1, data_res_.10_step_2, data_res_.25_step_.5, data_res_.25_step_1, data_res_.25_step_2)
# filter quality
data = data[data$qc_flag == 0,]
# filter doy
data = data[data$doy == 45,]
data$snow_depth_m = data$snow_depth_cm/100


# define standard error
sefx = function(data){
  sd(data, na.rm=TRUE)/sqrt(length(na.omit(data)))
}
# define rmse
rmse = function(difdata){
  sqrt(sum(difdata^2, na.rm = TRUE))/sqrt(length(na.omit(difdata)))
}
# define rmse
mae = function(difdata){
  sum(abs(difdata), na.rm = TRUE)/length(na.omit(difdata))
}

# compute/plot by cover type ####

data_rmse <- data %>% 
  gather(offset, DSM_snow_depth, 18:21, factor_key=FALSE) %>%
  mutate(hs_error_m = snow_depth_m - DSM_snow_depth) %>%
  group_by(res, step, offset, cover) %>%
  summarise(rmse_hs = rmse(hs_error_m), mae_hs = mae(hs_error_m), n = length(na.omit(hs_error_m)))

# offset as numeric
data_rmse$step = as.numeric(data_rmse$step)
data_rmse$offset = as.factor(data_rmse$offset)
data_rmse$res = as.factor(data_rmse$res)

ggplot(data_rmse, aes(x=step, y=rmse_hs, color=offset, shape=offset)) +
  facet_grid(res ~ cover) +
  geom_line() + 
  geom_point(size=2) +
  labs(title ='RMSE of LiDAR snow depth (HS) with ground samples across cover types, step, offset, and resolution', x = "Step (m)", y = "Snow Depth RMSE (m)", color = "Offset (m)", shape = "Offset (m)")

ggplot(data_rmse, aes(x=step, y=mae_hs, color=offset, shape=offset)) +
  facet_grid(res ~ cover) +
  geom_line() + 
  geom_point(size=2)

ggplot(data_rmse, aes(x=step, y=rmse_hs, color=res, shape=res)) +
  facet_grid(offset ~ cover) +
  geom_line() + 
  geom_point() +
  labs(title ='RMSE of LiDAR snow depth (HS) with ground samples across cover types, step, offset, and resolution', x = "Step (m)", y = "Snow Depth RMSE (m)", color = "Resolution (m)", shape = "Resolution (m)")

ggplot(data_rmse, aes(x=step, y=n, color=res, shape=offset)) +
  facet_grid(. ~ cover) +
  geom_line() + 
  geom_point(size=2) +
  labs(title ='Ground sample coverage of LiDAR snow depth across cover types, step, offset, and resolution', x = "Step (m)", y = "Samples (n)", color = "Resolution (m)", shape = "Offset (m)")
  
# compute/plot general ####

# generalized for all cover types
data_rmse_gen <- data %>% 
  gather(offset, DSM_snow_depth, 18:21, factor_key=FALSE) %>%
  mutate(hs_error_m = snow_depth_m - DSM_snow_depth) %>%
  group_by(res, step, offset) %>%
  summarise(rmse_hs = rmse(hs_error_m), n = length(na.omit(hs_error_m)))

# offset as numeric

data_rmse_gen$res = as.factor(data_rmse_gen$res)

ggplot(data_rmse_gen, aes(x=step, y=rmse_hs, color=res, shape=res)) +
  facet_grid(. ~ offset) + 
  geom_line() + 
  geom_point(size=2)

data_rmse_gen$step = as.factor(data_rmse_gen$step)
data_rmse_gen$offset = as.numeric(data_rmse_gen$offset)

ggplot(data_rmse_gen, aes(x=step, y=n, color=res, shape=res)) +
  facet_grid(. ~ offset) +
  geom_line() + 
  geom_point(size=2)
