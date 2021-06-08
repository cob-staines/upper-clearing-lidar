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

df = df %>%
  mutate(tx_error = synth_transmission - transmission, cn_error = -log(synth_transmission) - -log(transmission))

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

wrmse = function(difdata, weights){
  weights = weights / sum(weights, na.rm = TRUE)  # normalize
  sqrt(sum(weights * (difdata^2), na.rm = TRUE))
}

wmb = function(difdata, weights){
  weights = weights / sum(weights, na.rm = TRUE)  # normalize
  sum(weights * difdata, na.rm=TRUE)
}

mae = function(difdata){
  sum(abs(difdata), na.rm = TRUE)/length(na.omit(difdata))
}

# plots

# error metrics
df_agg = df %>%
  mutate(solid_angle = 2* pi * (cos((as.numeric(ring_number) - 1) * 15 * pi / 180) - cos(as.numeric(ring_number) * 15 * pi / 180))) %>%
  group_by(poisson_radius_m, optimization_scalar) %>%
  summarise(tx_rmse = rmse(tx_error), tx_mae = mae(tx_error), tx_mean_bias = mean(tx_error, na.rm = TRUE), tx_wrmse=wrmse(tx_error, solid_angle), tx_wmb = wmb(tx_error, solid_angle), cn_rmse = rmse(cn_error), cn_mae = mae(cn_error), cn_mean_bias = mean(cn_error, na.rm = TRUE), cn_wrmse=wrmse(cn_error, solid_angle), cn_wmb = wmb(cn_error, solid_angle))
df_agg$poisson_radius_m = as.factor(df_agg$poisson_radius_m)
# df_agg$optimization_scalar = as.numeric(df_agg$optimization_scalar)

# mean bias tx
ggplot(df_agg, aes(x=optimization_scalar, y=tx_mean_bias, color=poisson_radius_m)) +
  geom_point() +
  geom_line() +
  labs(x="point size scalar [-]", y="transmittance mean bias [-]")
# ggsave(paste0(plot_out_dir, "point_size_optimization_mean_bian.png"), width=p_width, height=p_height, dpi=dpi)

# weighted mean bias tx
df_agg %>%
  filter(poisson_radius_m == 0) %>%
ggplot(., aes(x=optimization_scalar, y=tx_wmb)) +
  geom_point() +
  geom_line() +
  labs(x="point size scalar [-]", y="transmittance mean bias [-]")
ggsave(paste0(plot_out_dir, "point_size_optimization_weighted_mean_bian.png"), width=p_width, height=p_height, dpi=dpi)

# weighted mean bias cn
ggplot(df_agg, aes(x=optimization_scalar, y=cn_wmb, color=poisson_radius_m)) +
  geom_point() +
  geom_line() +
  xlim(0, 1) +
  labs(x="point size scalar [-]", y="transmittance mean bias weighted by solid angle [-]")


# rmse tx
ggplot(df_agg, aes(x=optimization_scalar, y=tx_rmse, color=poisson_radius_m)) +
  geom_point() +
  geom_line() + 
  ylim(0, NA) +
  labs(x="point size scalar [-]", y="transmittance RMSE [-]")
# ggsave(paste0(plot_out_dir, "point_size_optimization_rmse.png"), width=p_width, height=p_height, dpi=dpi)

# wrmse tx
df_agg %>%
  filter(poisson_radius_m == 0) %>%
ggplot(., aes(x=optimization_scalar, y=tx_wrmse)) +
  geom_point() +
  geom_line() + 
  # ylim(0, NA) +
  labs(x="point size scalar [-]", y="transmittance RMSE [-]")
ggsave(paste0(plot_out_dir, "point_size_optimization_weighted_rmse.png"), width=p_width, height=p_height, dpi=dpi)

# wrmse cn
ggplot(df_agg, aes(x=optimization_scalar, y=cn_wrmse, color=poisson_radius_m)) +
  geom_point() +
  geom_line() + 
  ylim(0, NA) +
  xlim(0, 1.2) +
  labs(x="point size scalar [-]", y="transmittance RMSE weighted by solid angle [-]")

# mae tx
ggplot(df_agg, aes(x=optimization_scalar, y=tx_mae, color=poisson_radius_m)) +
  geom_point() +
  geom_line()

###
df %>%
  filter(poisson_radius_m == 0, optimization_scalar == 1.06) %>%
  ggplot(., aes(x=synth_transmission, y=transmission, color=ring_number)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1) +
  ylim(0, 1) +
  xlim(0, 1) +
  labs(title="Light transmittance (T) validation", x='T (point reprojection)', y='T (hemispherical photography)', color='Zenith angle\nband [deg]') +
  scale_color_discrete(labels = c("0-15", "15-30", "30-45", "45-60", "60-75"), breaks=c(1, 2, 3, 4, 5))
#ggsave(paste0(plot_out_dir, "snow_off_tx_error_eval.png"), width=p_width, height=p_height, dpi=dpi)
ggsave(paste0(plot_out_dir, "point_reprojection_tx_error_eval_os1.06.png"), width=p_width, height=p_height, dpi=dpi)

###

df_sub = df_agg %>%
  filter(poisson_radius_m == 0)
approx(x=df_sub$tx_mean_bias, y=df_sub$optimization_scalar, xout=0)
approx(x=df_sub$tx_wmb, y=df_sub$optimization_scalar, xout=0)

os = df_sub$optimization_scalar[1:(length(df_sub)-2)] + diff(df_sub$optimization_scalar) / 2
xx = diff(df_sub$tx_wrmse) / diff(df_sub$optimization_scalar)
approx(x=xx, y=os, xout=0)

df_sub = df_agg %>%
  filter(poisson_radius_m == 0.05)
approx(x=df_sub$tx_mean_bias, y=df_sub$optimization_scalar, xout=0)

df_sub = df_agg %>%
  filter(poisson_radius_m == 0.15)
approx(x=df_sub$tx_mean_bias, y=df_sub$optimization_scalar, xout=0)

# stats

x = -log(df$synth_transmission)
y = -log(df$transmission)
summary(lm(y ~ 0 + x))

x = df$synth_transmission
y = df$transmission
summary(lm(y ~ 0 + x))

model_eval = function(nn = 0){
  df_sub = df %>%
    filter(poisson_radius_m == 0, optimization_scalar == .66) %>%
    mutate(synth_cn = -log(synth_transmission), cn = -log(transmission), cn_error = synth_cn - cn)

  if(nn > 0){
    df_sub = df_sub %>%
      filter(ring_number == nn)
  }
  
  
  tx_ssres = sum((df_sub$tx_error)^2, na.rm=TRUE)
  tx_sstot = sum((mean(df_sub$transmission, na.rm=TRUE) - df_sub$transmission)^2, na.rm=TRUE)
  tx_r2 = 1 - tx_ssres / tx_sstot
  tx_r2_adj = 1 - (1 - tx_r2) * (sum(!is.na(df_sub$transmission)) - 1) / (sum(!is.na(df_sub$transmission)) - 2)
  
  cn_ssres = sum((df_sub$cn_error)^2, na.rm=TRUE)
  cn_sstot = sum((mean(df_sub$cn, na.rm=TRUE) - df_sub$cn)^2, na.rm=TRUE)
  cn_r2 = 1 - cn_ssres / cn_sstot
  cn_r2_adj = 1 - (1 - cn_r2) * (sum(!is.na(df_sub$cn)) - 1) / (sum(!is.na(df_sub$cn)) - 2)
  
  # stats
  rmse_cn = sqrt(mean((df_sub$cn_error)^2, na.rm=TRUE))
  rmse_tx = sqrt(mean((df_sub$tx_error)^2, na.rm=TRUE))
  
  c(tx_r2_adj, rmse_tx, cn_r2_adj, rmse_cn)
}

model_eval(1)
model_eval(2)
model_eval(3)
model_eval(4)
model_eval(5)

model_eval = function(nn = 0){
  
  df_sub = df %>%
    filter(poisson_radius_m == 0, optimization_scalar == .82)
  
  if(nn > 0){
    df_sub = df_sub %>%
      filter(ring_number == nn)
  }
  
  tx = df_sub$transmission
  cn = -log(tx)
  syn_tx = df_sub$synth_transmission
  syn_cn = -log(syn_tx)
  
  tx_lm = lm(tx ~ syn_tx)
  tx_r2_adj = summary(tx_lm)$adj.r.squared
  tx_rmse = summary(tx_lm)$sigma
  
  cn_lm = lm(cn ~ syn_cn)
  cn_r2_adj = summary(cn_lm)$adj.r.squared
  cn_rmse = summary(cn_lm)$sigma
  
  c(tx_r2_adj, tx_rmse, cn_r2_adj, cn_rmse)
}

model_eval(0)
model_eval(1)
model_eval(2)
model_eval(3)
model_eval(4)
model_eval(5)

ggplot(df_sub, aes(x=transmission, y=synth_transmission)) +
  geom_point()
