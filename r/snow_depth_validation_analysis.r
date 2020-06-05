# validation for snow_depth

library('dplyr')
library('tidyr')
library('ggplot2')

#load lidar samples

hs_19_045_in <- "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/products/hs/19_045/all_ground_points_hs_19_045.csv"
hs_19_050_in <- "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/products/hs/19_050/all_ground_points_hs_19_050.csv"
hs_19_052_in <- "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/products/hs/19_052/all_ground_points_hs_19_052.csv"
hs_19_107_in <- "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/products/hs/19_107/all_ground_points_hs_19_107.csv"
hs_19_123_in <- "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/products/hs/19_123/all_ground_points_hs_19_123.csv"

hs_19_045 = read.csv(hs_19_045_in, header=TRUE, na.strings = c("NA",""), sep=",")
colnames(hs_19_045)[13:17] = c("r.04", "r.10", "r.25", "r.50", "r1.00")
hs_19_045[,13:17][hs_19_045[,13:17] == -9999] = NA
hs_19_045$Date.Time = as.POSIXct(as.character(hs_19_045$Date.Time),format="%d/%m/%Y %H:%M:%S")
hs_19_045$survey_doy = floor(julian(hs_19_045$Date.Time, origin=as.POSIXct("2018-12-31")))
# filter to survey doy
hs_19_045 = hs_19_045[hs_19_045$survey_doy == 45,]

hs_19_050 = read.csv(hs_19_050_in, header=TRUE, na.strings = c("NA",""), sep=",")
colnames(hs_19_050)[13:17] = c("r.04", "r.10", "r.25", "r.50", "r1.00")
hs_19_050[,13:17][hs_19_050[,13:17] == -9999] = NA
hs_19_050$Date.Time = as.POSIXct(as.character(hs_19_050$Date.Time),format="%d/%m/%Y %H:%M:%S")
hs_19_050$survey_doy = floor(julian(hs_19_050$Date.Time, origin=as.POSIXct("2018-12-31")))
# filter to survey doy
hs_19_050 = hs_19_050[hs_19_050$survey_doy == 50,]

hs_19_052 = read.csv(hs_19_052_in, header=TRUE, na.strings = c("NA",""), sep=",")
colnames(hs_19_052)[13:17] = c("r.04", "r.10", "r.25", "r.50", "r1.00")
hs_19_052[,13:17][hs_19_052[,13:17] == -9999] = NA
hs_19_052$Date.Time = as.POSIXct(as.character(hs_19_052$Date.Time),format="%d/%m/%Y %H:%M:%S")
hs_19_052$survey_doy = floor(julian(hs_19_052$Date.Time, origin=as.POSIXct("2018-12-31")))
# filter to survey doy
hs_19_052 = hs_19_052[hs_19_052$survey_doy == 52,]

hs_19_107 = read.csv(hs_19_107_in, header=TRUE, na.strings = c("NA",""), sep=",")
colnames(hs_19_107)[13:17] = c("r.04", "r.10", "r.25", "r.50", "r1.00")
hs_19_107[,13:17][hs_19_107[,13:17] == -9999] = NA
hs_19_107$Date.Time = as.POSIXct(as.character(hs_19_107$Date.Time),format="%d/%m/%Y %H:%M:%S")
hs_19_107$survey_doy = floor(julian(hs_19_107$Date.Time, origin=as.POSIXct("2018-12-31")))
# filter to survey doy
hs_19_107 = hs_19_107[hs_19_107$survey_doy == 107,]

hs_19_123 = read.csv(hs_19_123_in, header=TRUE, na.strings = c("NA",""), sep=",")
colnames(hs_19_123)[13:17] = c("r.04", "r.10", "r.25", "r.50", "r1.00")
hs_19_123[,13:17][hs_19_123[,13:17] == -9999] = NA
hs_19_123$Date.Time = as.POSIXct(as.character(hs_19_123$Date.Time),format="%d/%m/%Y %H:%M:%S")
hs_19_123$survey_doy = floor(julian(hs_19_123$Date.Time, origin=as.POSIXct("2018-12-31")))
# filter to survey doy
hs_19_123 = hs_19_123[hs_19_123$survey_doy == 123,]

#compile
hs_data = rbind (hs_19_045, hs_19_050, hs_19_052, hs_19_107, hs_19_123)

#load snow survey samples
survey_in = "C:/Users/Cob/index/educational/usask/research/masters/data/surveys/depth_swe/snow_survey_gnss_merged.csv"
hs_survey = read.csv(survey_in, header=TRUE, na.strings = c("NA",""), sep=",")
hs_survey$snow_depth_m = hs_survey$snow_depth_cm/100
hs_survey = hs_survey[,c("uid","snow_depth_m")]

data = merge(hs_data, hs_survey, by="uid")

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
mbias = function(difdata){
  mean(difdata, na.rm = TRUE)
}

data_gather <- data %>% 
  gather(resolution, dsm_snow_depth, 13:17, factor_key=FALSE)
data_gather$dsm_snow_depth[data_gather$dsm_snow_depth < 0] = NA

data_summary = data_gather %>%
  mutate(hs_error_m = snow_depth_m - dsm_snow_depth) %>%
  group_by(resolution, cover) %>%
  summarise(rmse_hs = rmse(hs_error_m), mae_hs = mae(hs_error_m), mb_hs = mbias(hs_error_m), n = length(na.omit(hs_error_m)))

data_summary$survey_doy = as.numeric(data_summary$survey_doy)

# plots

ggplot(data_gather, aes(x=snow_depth_m, y=dsm_snow_depth, color=resolution)) +
  geom_point()

ggplot(data_summary, aes(x=cover, y=mb_hs, color=resolution, shape = resolution)) +
  geom_point()
ggplot(data_summary, aes(x=cover, y=rmse_hs, color=resolution, shape = resolution)) +
  geom_point()