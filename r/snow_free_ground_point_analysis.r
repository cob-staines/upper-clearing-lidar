library('dplyr')
library('tidyr')
library('ggplot2')
workingdir <- "C:/Users/Cob/index/educational/usask/research/masters/data/surveys/"

master <- "all_ground_points.csv"
dem <- "all_ground_points_dem.csv"
out <- "all_ground_points_diff.csv"

data <- read.csv(paste0(workingdir,master), header=TRUE, na.strings = c("NA",""), sep=",")
data$Date.Time <- as.POSIXct(data$Date.Time, format="%d/%m/%Y %H:%M:%OS")

dema <- read.csv(paste0(workingdir,dem), header=TRUE, na.strings = c("NA",""), sep=",")

diff <- data$WGS84.Ellip..Height - dema 
output <- cbind(data, diff) %>%
  filter(Point.Role == 'GNSSPhaseMeasuredRTK')

write.csv(output, paste0(workingdir,out), na = "")

all <- data.frame(c(data,dema)) %>%
  filter(Point.Role == 'GNSSPhaseMeasuredRTK')

analysis <- all %>%
  summarise(gnss_3d_error_mean_m = mean(CQ.3D),
            dh_.04_bias_m = mean(dem_.04m - WGS84.Ellip..Height, na.rm=TRUE),
            dh_.04_rmse_m = sqrt(mean((dem_.04m - WGS84.Ellip..Height)^2, na.rm=TRUE)/sum(!is.na(all$dem_.04m))),
            dh_.04_count = sum(!is.na(all$dem_.04m)),
            dh_.10_bias_m = mean(dem_.10m - WGS84.Ellip..Height, na.rm=TRUE),
            dh_.10_rmse_m = sqrt(mean((dem_.10m - WGS84.Ellip..Height)^2, na.rm=TRUE)/sum(!is.na(all$dem_.10m))),
            dh_.10_count = sum(!is.na(all$dem_.10m)),
            dh_.25_bias_m = mean(dem_.25m - WGS84.Ellip..Height, na.rm=TRUE),
            dh_.25_rmse_m = sqrt(mean((dem_.25m - WGS84.Ellip..Height)^2, na.rm=TRUE)/sum(!is.na(all$dem_.25m))),
            dh_.25_count = sum(!is.na(all$dem_.25m)),
            dh_.50_bias_m = mean(dem_.50m - WGS84.Ellip..Height, na.rm=TRUE),
            dh_.50_rmse_m = sqrt(mean((dem_.50m - WGS84.Ellip..Height)^2, na.rm=TRUE)/sum(!is.na(all$dem_.50m))),
            dh_.50_count = sum(!is.na(all$dem_.50m))
            )

bias_plot <- analysis %>%
  select(dh_.04_bias_m, dh_.10_bias_m, dh_.25_bias_m, dh_.50_bias_m) %>%
  gather(key="method", value="height_m")

ggplot(bias_plot, aes(x=method, y=height_m)) + 
  geom_bar(stat="identity")

rmse_plot <- analysis %>%
  select(dh_.04_rmse_m, dh_.10_rmse_m, dh_.25_rmse_m, dh_.50_rmse_m) %>%
  gather(key="method", value="height_m")

ggplot(rmse_plot, aes(x=method, y=height_m)) + 
  geom_bar(stat="identity")

ggplot(all, aes()) + 
  geom_point(aes(dem_.04m, WGS84.Ellip..Height, color = Point.Role)) + 
  geom_errorbar(aes(dem_.04m, ymax=WGS84.Ellip..Height + CQ.3D/2, ymin=WGS84.Ellip..Height - CQ.3D/2, width=.2)) +
  geom_abline(slope=1, intercept=0)

ggplot(all, aes()) + 
  geom_point(aes(WGS84.Ellip..Height, dem_.04m, color = '.04m')) +
  geom_point(aes(WGS84.Ellip..Height, dem_.10m, color = '.10m')) +
  geom_point(aes(WGS84.Ellip..Height, dem_.25m, color = '.25m')) +
  geom_point(aes(WGS84.Ellip..Height, dem_.50m, color = '.50m')) +
  geom_abline(slope=1, intercept=0)
