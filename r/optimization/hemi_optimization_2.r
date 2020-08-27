library('dplyr')
library('tidyr')
library('ggplot2')

photos_lai_in = "C:/Users/Cob/index/educational/usask/research/masters/data/hemispheres/19_149/clean/sized/LAI_parsed.csv"
photos_lai = read.csv(photos_lai_in, header=TRUE, na.strings = c("NA",""), sep=",")
photos_lai$original_file = gsub("_r.JPG", ".JPG", photos_lai$picture)
photos_meta_in = "C:/Users/Cob/index/educational/usask/research/masters/data/hemispheres/hemi_lookup_cleaned.csv"
photos_meta = read.csv(photos_meta_in, header=TRUE, na.strings = c("NA",""), sep=",")
photos_meta$id <- as.numeric(rownames(photos_meta)) - 1

photos = merge(photos_lai, photos_meta, by.x='original_file', by.y='filename', all.x=TRUE)

synth_lai_in = "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/synthetic_hemis/opt/poisson/LAI_parsed.csv"
synth_lai = read.csv(synth_lai_in, header=TRUE, na.strings = c("NA",""), sep=",")
synth_meta_in = "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/synthetic_hemis/opt/poisson/hemimetalog.csv"
synth_meta = read.csv(synth_meta_in, header=TRUE, na.strings = c("NA",""), sep=",")

synth = merge(synth_lai, synth_meta, by.x='picture', by.y='file_name', all.x=TRUE)

all = merge(synth, photos, by='id', all.x=TRUE, suffixes=c("_synth", "_photo"))

# calculate errors
all = all %>%
  mutate(lai_no_cor_error = lai_no_cor_synth - lai_no_cor_photo)

rmse = function(difdata){
  sqrt(sum(difdata^2, na.rm = TRUE))/sqrt(length(na.omit(difdata)))
}
# define rmse
mae = function(difdata){
  sum(abs(difdata), na.rm = TRUE)/length(na.omit(difdata))
}

all_agg = all %>%
  group_by(poisson_radius_m, optimization_scalar) %>%
  summarise(rmse_lai = rmse(lai_no_cor_error), mae_lai = mae(lai_no_cor_error), mean_bias_lai = mean(lai_no_cor_error, na.rm = TRUE) , n = length(na.omit(lai_no_cor_error)))

### BOOKMARK -- everything below is old hat, needs addaptation.

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
  
  