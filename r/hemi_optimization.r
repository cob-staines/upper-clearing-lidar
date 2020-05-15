library('dplyr')
library('tidyr')
library('ggplot2')

photos_in = "C:/Users/Cob/index/educational/usask/research/masters/data/hemispheres/19_149/clean/sized/photo_LAI.dat"
photos = read.csv(photos_in, header=TRUE, na.strings = c("NA",""), sep="")
colnames(photos)[1] = "picture_file"
photos = gather(photos, method, lai, 4:5)
photos$lai = as.numeric(sub(",", ".", photos$lai, fixed = TRUE))
photos$corrections = levels(photos$corrections) = c("", "S_al_2007")


prelim_synth_in = "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/synthetic_hemis/opt/var_os_white_bg/LAI.dat"
prelim = read.csv(prelim_synth_in, header=TRUE, na.strings = c("NA",""), sep="")
colnames(prelim)[1] = "picture_file"
prelim = gather(prelim, method, lai, 4:5)
prelim$lai = as.numeric(sub(",", ".", prelim$lai, fixed = TRUE))
prelim$corrections = levels(prelim$corrections) = c("", "S_al_2007")

synth_0.05_in = "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/synthetic_hemis/opt/os_0.05/white_bg/LAI.dat"
synth_0.05 = read.csv(synth_0.05_in, header=TRUE, na.strings = c("NA",""), sep="")
synth_0.1_in = "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/synthetic_hemis/opt/os_0.1/white_bg/LAI.dat"
synth_0.1 = read.csv(synth_0.1_in, header=TRUE, na.strings = c("NA",""), sep="")
synth_0.25_in = "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/synthetic_hemis/opt/os_0.25/white_bg/LAI.dat"
synth_0.25 = read.csv(synth_0.25_in, header=TRUE, na.strings = c("NA",""), sep="")

synth = rbind(synth_0.05, synth_0.1, synth_0.25)

colnames(synth)[1] = "picture_file"
synth = gather(synth, method, lai, 4:5)
synth$lai = as.numeric(sub(",", ".", synth$lai, fixed = TRUE))
synth$corrections = levels(synth$corrections) = c("", "S_al_2007")

# break out strings for photo names
photos$loc = gsub("_r.JPG", "", as.character(photos$picture_file))

name_parse <- function(name){
  a = strsplit(as.character(name), "_img_")
  las = gsub("las_", "", a[[1]][1])
  b = strsplit(a[[1]][2], "_os_")
  loc = b[[1]][1]
  os = gsub(".png", "", b[[1]][2])
  return(c(las, loc, os))
}
col_parse <- function(col){
  colen = length(col)
  loc = rep(NA, colen)
  las = rep(NA, colen)
  os = rep(NA, colen)
  
  col = lapply(col, as.character)
  
  for (ii in 1:colen){
    temp = name_parse(col[ii])
    las[ii] <- temp[1]
    loc[ii] <- temp[2]
    os[ii] <- temp[3]
  }
  peace = data.frame(las, loc, os)
  return(peace)
}

prelim = cbind(prelim, col_parse(prelim$picture_file))
synth = cbind(synth, col_parse(synth$picture_file))


# merge on location, method, correciton
prelim_photos = merge(prelim, photos, by=c("loc", "method", "corrections"), all.x=TRUE, suffixes=c("_las", "_photo"))
synth_photos = merge(synth, photos, by=c("loc", "method", "corrections"), all.x=TRUE, suffixes=c("_las", "_photo"))

prelim_photos$os = as.numeric(as.character(prelim_photos$os))
prelim_photos$th_hold_las = as.factor(prelim_photos$th_hold_las)


# calculate errors
prelim_photos = prelim_photos %>%
  mutate(lai_error = lai_las - lai_photo, group = paste0(method, "_", corrections))

synth_photos = synth_photos %>%
  mutate(lai_error = lai_las - lai_photo, group = paste0(method, "_", corrections))

rmse = function(difdata){
  sqrt(sum(difdata^2, na.rm = TRUE))/sqrt(length(na.omit(difdata)))
}
# define rmse
mae = function(difdata){
  sum(abs(difdata), na.rm = TRUE)/length(na.omit(difdata))
}

synth_photos_da = synth_photos %>%
  group_by(method, corrections, os, th_hold_las, group) %>%
  summarise(rmse_lai = rmse(lai_error), mae_lai = mae(lai_error), mean_bias_lai = mean(lai_error, na.rm = TRUE) , n = length(na.omit(lai_error)))

# plot
# RMSE
ggplot(synth_photos_da, aes(x=th_hold_las, y=rmse_lai, color=os, shape = group)) +
  geom_point() +
  geom_line() +
  labs(title="RMSE of LAI across point size, threshold, and methods", x="Synthetic threshold (1-255)", y="RMSE of LAI (LiDAR vs. Photo)", color="Synthetic point size", shape="LAI method")

# mean_bias
ggplot(synth_photos_da, aes(x=th_hold_las, y=mean_bias_lai, color=os, shape=group)) +
  geom_point() +
  geom_line() +
  labs(title="Mean Bias of LAI across point size, threshold, and methods", x="Synthetic threshold (1-255)", y="Mean Bias of LAI (LiDAR - Photo)", color="Synthetic point size", shape="LAI method")
  

synth_photos_da$th_hold_las = as.factor(synth_photos_da$th_hold_las)
synth_photos_da$os = as.numeric(as.character(synth_photos_da$os))

ggplot(synth_photos_da, aes(x=os, y=mean_bias_lai, color=th_hold_las, shape=group)) +
  geom_point() +
  geom_line()


# what kind of spread do we see amonth the primary contenders?

synth_photos$th_hold_las = as.factor(synth_photos$th_hold_las)
synth_photos$os = as.factor(synth_photos$os)

selection_1 <- synth_photos %>%
  filter(os==0.05, th_hold_las==128, group=="Miller_")
selection_2 <- synth_photos %>%
  filter(os==0.1, th_hold_las==80, group=="Miller_")
selection_3 <- synth_photos %>%
  filter(os==0.25, th_hold_las==32, group=="Miller_")

selection = rbind(selection_1, selection_2, selection_3)

ggplot(selection, aes(x=lai_photo, y=lai_las, color=os, label=loc)) + 
  geom_point() +
  geom_abline(slope=1, intercept=0) +
  labs(title="Photo-LAI vs. LiDAR-LAI for 3 best cases", x="Photo LAI", y="LiDAR LAI", color="Synthetic point size")

ggplot(synth_photos, aes(x=lai_photo, y=lai_las, color=th_hold_las)) + 
  facet_grid(os ~ group) +
  geom_point() +
  geom_abline(slope=1, intercept=0)
  
  