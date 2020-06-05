library('dplyr')
library('tidyr')
library('ggplot2')

pnts_in <- "C:/Users/Cob/index/educational/usask/research/masters/data/surveys/all_ground_points_UTM11N_uid.csv"
data.25_in = "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/19_149/19_149_all_200311/OUTPUT_FILES/DEM/19_149_all_200311_628000_5646525dem_.25m_ground_pount_samples.csv"
data.04_in = "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/19_149/19_149_all_200311/OUTPUT_FILES/DEM/19_149_all_200311_628000_5646525dem_.04m_ground_pount_samples.csv"

#import all points
pnts = read.csv(pnts_in, header=TRUE, na.strings = c("NA",""), sep=",")

#import points from .04 and .25 res file
data.25 <- read.csv(data.25_in, header=TRUE, na.strings = c("NA",""), sep=",")
data.04 <- read.csv(data.04_in, header=TRUE, na.strings = c("NA",""), sep=",")
# all numeric data together
difdata = pnts$WGS84.Ell - data.frame(data.25[10:16], data.04[10:16])

# calculate standard error of rows
sefx = function(data){
  sd(data, na.rm=TRUE)/sqrt(length(na.omit(data)))
}
dif_se = apply(difdata,1,sefx)
dif_mean = rowMeans(difdata, na.rm = TRUE)

summary_data = data.frame(pnts, dif_se, dif_mean)

# plot standard-error with mean
ggplot(summary_data, aes(x=dif_mean, y=dif_se, color=Point.Rol)) +
  geom_point() + 
  geom_text(aes(label=uid),hjust=0, vjust=0)

# filter points: drop phase-measured and hand-selected
pnts_filter = (pnts$Point.Rol == "GNSSPhaseMeasuredRTK") & (pnts$uid != 164) & (pnts$uid != 154) & (pnts$uid != 98) & (pnts$uid != 158) & (pnts$uid != 100) & (pnts$uid != 95)

sum_data_filtered = summary_data[pnts_filter,]

# plot filtered standard-error with mean
ggplot(sum_data_filtered, aes(x=dif_mean, y=dif_se, color=Point.Rol)) +
  geom_point() + 
  geom_text(aes(label=uid),hjust=0, vjust=0)

# plot standard error of each data set using filtered points
set_se = apply(difdata[pnts_filter,],2,sefx)
set = colnames(difdata)

step_sa = data.frame(set, set_se)
ggplot(step_sa, aes(x=set, y=set_se)) + 
  geom_point() + 
  labs(title ='Standard error of ground validation points for different step sizes and resolutions', x = "r[resolution]s[step] in m", y = "standard error with QC'd ground points")
# looks like step of 1 gives us the best results! great work Cob!

step_sa_long = gather(difdata[pnts_filter,], resstep, discrepancy, 1:14, factor_key=TRUE)
ggplot(step_sa_long, aes(x=resstep, y=discrepancy)) + 
  geom_point()

# looking at the points themselves, we see that a smaller step size tends to over-estimate ground surface elevation (likely near edges?)
ggplot(step_sa_long, aes(x=resstep, y=discrepancy)) + 
  geom_boxplot()
# no significant bias noted