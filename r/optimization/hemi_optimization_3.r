library('dplyr')
library('tidyr')
library('ggplot2')

# optimize by transmission over analysis rings

plot_out_dir = "C:/Users/Cob/index/educational/usask/research/masters/graphics/thesis_graphics/validation/hemiphoto_validation/"
p_width = 8  # inches
p_height = 5.7  # inches
dpi = 100

photos_lai_in = "C:/Users/Cob/index/educational/usask/research/masters/data/hemispheres/19_149/clean/sized/thresholded/LAI_parsed.dat"
photos_lai = read.csv(photos_lai_in, header=TRUE, na.strings = c("NA",""), sep=",")
photos_lai$original_file = gsub("_r.jpg", ".JPG", photos_lai$picture)
photos_meta_in = "C:/Users/Cob/index/educational/usask/research/masters/data/hemispheres/hemi_lookup_cleaned.csv"
photos_meta = read.csv(photos_meta_in, header=TRUE, na.strings = c("NA",""), sep=",")
photos_meta$id <- as.numeric(rownames(photos_meta)) - 1

photos = merge(photos_lai, photos_meta, by.x='original_file', by.y='filename', all.x=TRUE)
photos = photos[, c("id", "transmission_s_1", "transmission_s_2", "transmission_s_3", "transmission_s_4", "transmission_s_5")]


synth_lai_in = "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/synthetic_hemis/opt/poisson/LAI_parsed.dat"
synth_lai = read.csv(synth_lai_in, header=TRUE, na.strings = c("NA",""), sep=",")
synth_meta_in = "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/synthetic_hemis/opt/poisson/hemimetalog.csv"
synth_meta = read.csv(synth_meta_in, header=TRUE, na.strings = c("NA",""), sep=",")
synth_meta$footprint = sqrt(synth_meta$point_size_scalar / (synth_meta$optimization_scalar * 2834.64))

comp_times = synth_meta %>%
  group_by(poisson_radius_m) %>%
  summarise(ct_mean = mean(computation_time_s))

synth = merge(synth_lai, synth_meta, by.x='picture', by.y='file_name', all.x=TRUE)
synth = synth[, c("id", "poisson_radius_m", "optimization_scalar", "transmission_s_1", "transmission_s_2", "transmission_s_3", "transmission_s_4", "transmission_s_5")]
names(synth) = c("id", "poisson_radius_m", "optimization_scalar", "synth_transmission_s_1", "synth_transmission_s_2", "synth_transmission_s_3", "synth_transmission_s_4", "synth_transmission_s_5")
# all = merge(synth, photos, by='id', all.x=TRUE, suffixes=c("_synth", "_photo"))
df = merge(synth, photos, by='id', all.x=TRUE)

df = df %>%
  distinct() %>%  # because there are some duplicate rows here...
  gather(key, value, -c(id, poisson_radius_m, optimization_scalar)) %>%
  extract(key, c("val_type", "ring_number"), "(\\D+)_s_(\\d)") %>%
  spread(val_type, value)
# df$optimization_scalar = as.factor(df$optimization_scalar)

ggplot(df, aes(x=transmission, y=synth_transmission, color=optimization_scalar)) +
  facet_grid(poisson_radius_m ~ .) +
  geom_point()

df = df %>%
  mutate(tx_error = synth_transmission - transmission)

# # calculate errors
# all = all %>%
#   mutate(lai_no_cor_error = lai_no_cor_synth - lai_no_cor_photo) %>%
#   mutate(lai_s_cc_error = lai_s_cc_synth - lai_s_cc_photo) %>%
#   mutate(co_error = openness_synth - openness_photo) %>%
#   mutate(co_error_1 = transmission_s_1_synth - transmission_s_1_photo) %>%
#   mutate(co_error_2 = transmission_s_2_synth - transmission_s_2_photo) %>%
#   mutate(co_error_3 = transmission_s_3_synth - transmission_s_3_photo) %>%
#   mutate(co_error_4 = transmission_s_4_synth - transmission_s_4_photo) %>%
#   mutate(co_error_5 = transmission_s_5_synth - transmission_s_5_photo)

rmse = function(difdata){
  sqrt(sum(difdata^2, na.rm = TRUE))/sqrt(length(na.omit(difdata)))
}
# define rmse
mae = function(difdata){
  sum(abs(difdata), na.rm = TRUE)/length(na.omit(difdata))
}

# plots

# optimized LAI
df_agg = df %>%
  group_by(poisson_radius_m, optimization_scalar) %>%
  summarise(tx_rmse = rmse(tx_error), tx_mae = mae(tx_error), tx_mean_bias = mean(tx_error, na.rm = TRUE))
df_agg$poisson_radius_m = as.factor(df_agg$poisson_radius_m)
# df_agg$optimization_scalar = as.numeric(df_agg$optimization_scalar)

# mean bias tx
ggplot(df_agg, aes(x=optimization_scalar, y=tx_mean_bias, color=poisson_radius_m)) +
  geom_point() +
  geom_line() +
  labs(x="point size scalar [-]", y="transmittance mean bias [-]")
ggsave(paste0(plot_out_dir, "point_size_optimization_mean_bian.png"), width=p_width, height=p_height, dpi=dpi)

# rmse tx
ggplot(df_agg, aes(x=optimization_scalar, y=tx_rmse, color=poisson_radius_m)) +
  geom_point() +
  geom_line() + 
  ylim(0, NA) +
  labs(x="point size scalar [-]", y="transmittance RMSE [-]")
ggsave(paste0(plot_out_dir, "point_size_optimization_rmse.png"), width=p_width, height=p_height, dpi=dpi)

# mae tx
ggplot(df_agg, aes(x=optimization_scalar, y=tx_mae, color=poisson_radius_m)) +
  geom_point() +
  geom_line()

df_sub = df_agg %>%
  filter(poisson_radius_m == 0)
approx(x=df_sub$tx_mean_bias, y=df_sub$optimization_scalar, xout=0)

df_sub = df_agg %>%
  filter(poisson_radius_m == 0.05)
approx(x=df_sub$tx_mean_bias, y=df_sub$optimization_scalar, xout=0)

df_sub = df_agg %>%
  filter(poisson_radius_m == 0.15)
approx(x=df_sub$tx_mean_bias, y=df_sub$optimization_scalar, xout=0)
