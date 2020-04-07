library('dplyr')
library('tidyr')
library('ggplot2')

data_step_.5_in <- "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/19_149/19_149_all_200311/OUTPUT_FILES/DEM/offset_opt/19_149_all_200311_628000_5646525dem_.04_step_.5_ground_pount_samples.csv"
data_step_1_in <- "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/19_149/19_149_all_200311/OUTPUT_FILES/DEM/offset_opt/19_149_all_200311_628000_5646525dem_.04_step_1_ground_pount_samples.csv"
data_step_2_in <- "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/19_149/19_149_all_200311/OUTPUT_FILES/DEM/offset_opt/19_149_all_200311_628000_5646525dem_.04_step_2_ground_pount_samples.csv"

#import all points
# step .5
data_step_.5 = read.csv(data_step_.5_in, header=TRUE, na.strings = c("NA",""), sep=",")
colnames(data_step_.5)[13:18] = c(".01", ".03", ".04", ".05", ".1", ".2")
data_step_.5$step = .5

# step 1
data_step_1 = read.csv(data_step_1_in, header=TRUE, na.strings = c("NA",""), sep=",")
data_step_1$step = 1
colnames(data_step_1)[13:18] = c(".01", ".03", ".04", ".05", ".1", ".2")
data_step_1$step = 1

# step 2
data_step_2 = read.csv(data_step_2_in, header=TRUE, na.strings = c("NA",""), sep=",")
data_step_2$step = 2
colnames(data_step_2)[13:18] = c(".01", ".03", ".04", ".05", ".1", ".2")
data_step_2$step = 2

data = rbind(data_step_.5, data_step_1, data_step_2)
# filter quality
data = data[data$qc_flag == 0,]

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

data_se <- data %>% 
  gather(offset, DSM_elev, 13:18, factor_key=FALSE) %>%
  mutate(error_elev = WGS84.Ell - DSM_elev) %>%
  group_by(step, offset, cover) %>%
  summarise(rmse_elev = rmse(error_elev), n = n())

# offset as numeric
data_se$step = as.factor(data_se$step)
data_se$offset = as.numeric(data_se$offset)

ggplot(data_se, aes(x=offset, y=rmse_elev, color=step)) +
  facet_grid(. ~ cover) +
  geom_line() + 
  geom_point() +
  labs(title ='RMSE of LiDAR DEM with GNSS points at .04m resolution for differenc cover types', x = "Offset (m)", y = "RMSE (m)", color = "Step (m)")

# generalized for all cover types
data_se_gen <- data %>% 
  gather(offset, DSM_elev, 13:18, factor_key=FALSE) %>%
  mutate(error_elev = WGS84.Ell - DSM_elev) %>%
  group_by(step, offset) %>%
  summarise(rmse_elev = rmse(error_elev))

# offset as numeric
data_se_gen$step = as.factor(data_se_gen$step)
data_se_gen$offset = as.numeric(data_se_gen$offset)

ggplot(data_se_gen, aes(x=offset, y=rmse_elev, group=step, color=step)) +
  geom_line() + 
  geom_point() +
  labs(title ='RMSE LiDAR DEM with GNSS points at .04m resolution', x = "Offset (m)", y = "RMSD (m)", color = "Step (m)")

# other ground points penalizing us?
point_se <- data %>%
  gather(offset, DSM_elev, 13:18, factor_key=FALSE) %>%
  mutate(error_elev = WGS84.Ell - DSM_elev) %>%
  group_by(uid, cover) %>%
  summarise(elev_dif_se = sefx(error_elev), elev_dif_bias = mean(error_elev))

ggplot(point_se, aes(x=elev_dif_bias, y=elev_dif_se, color=cover)) + 
  geom_point() +
  labs(title ='QCd Bare Elevation Error and Bias by ground point', x = "Elevation mean", y = "Elevation SE", color = "Step (m)")

