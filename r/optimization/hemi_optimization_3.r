library('dplyr')
library('tidyr')
library('ggplot2')

# optimize by transmission over analysis rings

plot_out_dir = "C:/Users/Cob/index/educational/usask/research/masters/graphics/thesis_graphics/validation/reprojection_validation/"
p_width = 8  # inches
p_height = 5.7  # inches
dpi = 100

photos_lai_in = "C:/Users/Cob/index/educational/usask/research/masters/data/hemispheres/19_149/clean/sized/thresholded/LAI_parsed.dat"
photos_lai = read.csv(photos_lai_in, header=TRUE, na.strings = c("NA",""), sep=",")
photos_lai$original_file = gsub("_r.jpg", ".JPG", photos_lai$picture, ignore.case=TRUE)
photos_meta_in = "C:/Users/Cob/index/educational/usask/research/masters/data/hemispheres/hemi_lookup_cleaned.csv"
photos_meta = read.csv(photos_meta_in, header=TRUE, na.strings = c("NA",""), sep=",")
photos_meta$id <- as.numeric(rownames(photos_meta)) - 1

photos = merge(photos_lai, photos_meta, by.x='original_file', by.y='filename', all.x=TRUE)
photos = photos[, c("id", "transmission_s_1", "transmission_s_2", "transmission_s_3", "transmission_s_4", "transmission_s_5")]


synth_lai_in = "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/synthetic_hemis/opt/flip_rerun/LAI_parsed.dat"
synth_lai = read.csv(synth_lai_in, header=TRUE, na.strings = c("NA",""), sep=",")
synth_meta_in = "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/synthetic_hemis/opt/flip_rerun/hemimetalog.csv"
synth_meta = read.csv(synth_meta_in, header=TRUE, na.strings = c("NA",""), sep=",")
synth_meta$footprint = sqrt(synth_meta$point_size_scalar / (synth_meta$optimization_scalar * 2834.64))

comp_times = synth_meta %>%
  group_by(poisson_radius_m) %>%
  summarise(ct_mean = mean(computation_time_s))

synth = merge(synth_lai, synth_meta, by.x='picture', by.y='file_name', all.x=TRUE)
# synth = synth[, c("id", "poisson_radius_m", "optimization_scalar", "transmission_s_1", "transmission_s_2", "transmission_s_3", "transmission_s_4", "transmission_s_5")]
synth = synth[, c("id", "poisson_radius_m", "optimization_scalar", "transmission_s_1", "transmission_s_2", "transmission_s_3", "transmission_s_4")]
# names(synth) = c("id", "poisson_radius_m", "optimization_scalar", "synth_transmission_s_1", "synth_transmission_s_2", "synth_transmission_s_3", "synth_transmission_s_4", "synth_transmission_s_5")
names(synth) = c("id", "poisson_radius_m", "optimization_scalar", "synth_transmission_s_1", "synth_transmission_s_2", "synth_transmission_s_3", "synth_transmission_s_4")
# all = merge(synth, photos, by='id', all.x=TRUE, suffixes=c("_synth", "_photo"))
df = merge(synth, photos, by='id', all.x=TRUE)

df = df %>%
  distinct() %>%  # because there are some duplicate rows here...
  gather(key, value, -c(id, poisson_radius_m, optimization_scalar)) %>%
  extract(key, c("val_type", "ring_number"), "(\\D+)_s_(\\d)") %>%
  spread(val_type, value)
# df$optimization_scalar = as.factor(df$optimization_scalar)

df = df %>%
  mutate(tx_error = synth_transmission - transmission, cn_error = -log(synth_transmission) - -log(transmission))%>%
  mutate(solid_angle = 2* pi * (cos((as.numeric(ring_number) - 1) * 15 * pi / 180) - cos(as.numeric(ring_number) * 15 * pi / 180)))

# drop ring 5
df_drop = df[df$ring_number != "5",]
# df_drop = df[df$ring_number == "1",]

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
df_agg = df_drop %>%
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
# ggsave(paste0(plot_out_dir, "point_size_optimization_tx_weighted_mean_bian.png"), width=p_width, height=p_height, dpi=dpi)

# weighted mean bias cn
df_agg %>%
  filter(poisson_radius_m == 0) %>%
ggplot(., aes(x=optimization_scalar, y=cn_wmb)) +
  geom_point() +
  geom_line() +
  xlim(0.2, NA) +
  ylim(-0.15, NA) +
  labs(x="point size scalar [-]", y="contact number mean bias weighted by solid angle [-]")
#ggsave(paste0(plot_out_dir, "point_size_optimization_cn_weighted_mean_bian.png"), width=p_width, height=p_height, dpi=dpi)


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
  # xlim(0, 0.21) +
  labs(x="point size scalar [-]", y="transmittance RMSE [-]")
# ggsave(paste0(plot_out_dir, "point_size_optimization_tx_weighted_rmse.png"), width=p_width, height=p_height, dpi=dpi)

# wrmse cn
df_agg %>%
  filter(poisson_radius_m == 0) %>%
ggplot(., aes(x=optimization_scalar, y=cn_wrmse)) +
  geom_point() +
  geom_line() +
  # xlim(0.2, NA) +
  # ylim(NA, 0.28) +
  labs(x="point size scalar [-]", y="RMSE weighted by solid angle [-]")
# ggsave(paste0(plot_out_dir, "point_size_optimization_cn_weighted_rmse.png"), width=p_width, height=p_height, dpi=dpi)

# mae tx
ggplot(df_agg, aes(x=optimization_scalar, y=tx_mae, color=poisson_radius_m)) +
  geom_point() +
  geom_line()

###

# cn plot
df_drop %>%
  filter(poisson_radius_m == 0, optimization_scalar == 1.01) %>%
ggplot(., aes(x=-log(synth_transmission), y=-log(transmission), color=ring_number)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1) +
  labs(title="Reprojection ontact number (X) validation", x='X (point cloud reprojection)', y='X (thresholded hemispherical photography)', color='Zenith angle\nband [deg]') +
  scale_color_discrete(labels = c("0-15", "15-30", "30-45", "45-60", "60-75"), breaks=c(1, 2, 3, 4, 5))
# ggsave(paste0(plot_out_dir, "point_reprojection_cn_error_eval_os1.01.png"), width=p_width, height=p_height, dpi=dpi)


# tx plot
df_drop %>%
  filter(poisson_radius_m == 0, optimization_scalar == 0.064) %>%
ggplot(., aes(x=synth_transmission, y=transmission, color=ring_number)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1) +
  ylim(0, 1) +
  xlim(0, 1) +
  labs(title="Reprojection light transmittance (T) validation", x='T (point cloud reprojection)', y='T (hemispherical photography)', color='Zenith angle\nband [deg]') +
  scale_color_discrete(labels = c("0-15", "15-30", "30-45", "45-60", "60-75"), breaks=c(1, 2, 3, 4, 5))
# ggsave(paste0(plot_out_dir, "point_reprojection_tx_error_eval_os1.01.png"), width=p_width, height=p_height, dpi=dpi)

# df %>%
#   filter(poisson_radius_m == 0, optimization_scalar == 1.3) %>%
#   ggplot(., aes(x=-log(synth_transmission), y=-log(transmission), color=ring_number)) +
#   geom_point()

###

df_sub = df_agg %>%
  filter(poisson_radius_m == 0)
approx(x=df_sub$tx_mean_bias, y=df_sub$optimization_scalar, xout=0)
approx(x=df_sub$cn_mean_bias, y=df_sub$optimization_scalar, xout=0)
approx(x=df_sub$tx_wmb, y=df_sub$optimization_scalar, xout=0)
approx(x=df_sub$cn_wmb, y=df_sub$optimization_scalar, xout=0)

os = df_sub$optimization_scalar[1:(nrow(df_sub)-1)] + diff(df_sub$optimization_scalar) / 2
xx = diff(df_sub$cn_wrmse) / diff(df_sub$optimization_scalar)
approx(x=xx, y=os, xout=0)

yy = summary(app)$coefficients[1] + xx * summary(app)$coefficients[2]
ox = -summary(app)$coefficients[1] / summary(app)$coefficients[2]

os = df_sub$optimization_scalar[1:(nrow(df_sub)-1)] + diff(df_sub$optimization_scalar) / 2
xx = diff(df_sub$tx_wrmse) / diff(df_sub$optimization_scalar)
approx(x=xx[1:6], y=os[1:6], xout=0, method="linear")

df_sub = df_agg %>%
  filter(poisson_radius_m == 0.05)
approx(x=df_sub$tx_mean_bias, y=df_sub$optimization_scalar, xout=0)

df_sub = df_agg %>%
  filter(poisson_radius_m == 0.15)
approx(x=df_sub$tx_mean_bias, y=df_sub$optimization_scalar, xout=0)

# stats

model_eval = function(nn = 0){
  
  df_sub = df_drop %>%
    filter(poisson_radius_m == 0, optimization_scalar == 1.01)
  
  if(nn > 0){
    df_sub = df_sub %>%
      filter(ring_number == nn)
  }
  
  tx = df_sub$transmission
  cn = -log(tx)
  syn_tx = df_sub$synth_transmission
  syn_cn = -log(syn_tx)
  weights = df_sub$solid_angle
  weights = weights / sum(weights)
  
  tx_error = syn_tx - tx
  cn_error = syn_cn - cn
  
  tx_lm = lm(tx ~ syn_tx, weights=weights)
  tx_r2_adj = summary(tx_lm)$adj.r.squared
  tx_rmse = summary(tx_lm)$sigma
  tx_p = pf(summary(tx_lm)$fstatistic[1], summary(tx_lm)$fstatistic[2], summary(tx_lm)$fstatistic[3], lower.tail=FALSE)
  
  tx_wrmse = sqrt(sum(weights * (tx_error^2)))
  tx_wmb = sum(tx_error * weights)
  
  cn_lm = lm(cn ~ syn_cn, weights=weights)
  cn_r2_adj = summary(cn_lm)$adj.r.squared
  cn_rmse = summary(cn_lm)$sigma
  cn_p = pf(summary(cn_lm)$fstatistic[1], summary(cn_lm)$fstatistic[2], summary(cn_lm)$fstatistic[3], lower.tail=FALSE)
  
  cn_wrmse = sqrt(sum(weights * (cn_error^2)))
  cn_wmb = sum(cn_error * weights)
  
  # c(tx_r2_adj, tx_p, tx_wrmse, tx_wmb, cn_r2_adj, cn_p, cn_wrmse, cn_wmb)
  c(tx_r2_adj, tx_wrmse, tx_wmb, cn_r2_adj, cn_wrmse, cn_wmb)
}

model_eval(0)
model_eval(1)
model_eval(2)
model_eval(3)
model_eval(4)

