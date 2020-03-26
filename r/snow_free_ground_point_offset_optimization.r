library('dplyr')
library('tidyr')
library('ggplot2')

data_step_.5_in <- "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/19_149/19_149_all_200311/OUTPUT_FILES/DEM/offset_opt/19_149_all_200311_628000_5646525dem_.04_step_.5_ground_pount_samples.csv"
data_step_1_in <- "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/19_149/19_149_all_200311/OUTPUT_FILES/DEM/offset_opt/19_149_all_200311_628000_5646525dem_.04_step_1_ground_pount_samples.csv"
data_step_2_in <- "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/19_149/19_149_all_200311/OUTPUT_FILES/DEM/offset_opt/19_149_all_200311_628000_5646525dem_.04_step_2_ground_pount_samples.csv"

# define standard error
sefx = function(data){
  sd(data, na.rm=TRUE)/sqrt(length(na.omit(data)))
}

# there is a better way to do all of this with dplyr. Do it. Tomorrow.

#import all points
# step .5
data_step_.5 = read.csv(data_step_.5_in, header=TRUE, na.strings = c("NA",""), sep=",")
data_step_.5 = data_step_.5[data_step_.5$qc_flag == 0,]
colnames(data_step_.5)[13:18] = c(".01", ".03", ".04", ".05", ".1", ".2")
dif_.5 = data_step_.5$WGS84.Ell - data_step_.5[13:18]

offset = as.numeric(colnames(data_step_.5)[13:18])
set_se = apply(dif_.5,2,sefx)
step = as.factor(.5)

data_se_.5 = data.frame(step, offset, set_se)

# step 1
data_step_1 = read.csv(data_step_1_in, header=TRUE, na.strings = c("NA",""), sep=",")
data_step_1 = data_step_1[data_step_1$qc_flag == 0,]
data_step_1$step = 1
colnames(data_step_1)[13:18] = c(".01", ".03", ".04", ".05", ".1", ".2")
dif_1 = data_step_1$WGS84.Ell - data_step_1[13:18]

offset = as.numeric(colnames(data_step_1)[13:18])
set_se = apply(dif_1,2,sefx)
step = as.factor(1)

data_se_1 = data.frame(step, offset, set_se)

# step 2
data_step_2 = read.csv(data_step_2_in, header=TRUE, na.strings = c("NA",""), sep=",")
data_step_2 = data_step_2[data_step_2$qc_flag == 0,]
data_step_2$step = 2
colnames(data_step_2)[13:18] = c(".01", ".03", ".04", ".05", ".1", ".2")
dif_2 = data_step_2$WGS84.Ell - data_step_2[13:18]

offset = as.numeric(colnames(data_step_2)[13:18])
set_se = apply(dif_2,2,sefx)
step = as.factor(2)

data_se_2 = data.frame(step, offset, set_se)

# compile

data_se_all = rbind(data_se_.5, data_se_1, data_se_2)

ggplot(data_se_all, aes(x = offset, y = set_se, color = step)) + 
  geom_line() +
  labs(title ='Standard error LiDAR DEM with GNSS points at .04m resolution', x = "Offset (m)", y = "Standard Error (m)")
# we see that step size of 1, offset of 0.04 minimizes the error amonth the tested sets.
# step size 1 out-performs all in the neighborhood of optimized offset
# we would like to run seperate analysis in forests and in clearing, to see if performance is optomized differently