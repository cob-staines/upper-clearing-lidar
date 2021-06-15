library('dplyr')
library('tidyr')
library('ggplot2')

plot_out_dir = "C:/Users/Cob/index/educational/usask/research/masters/graphics/thesis_graphics/validation/ray_sampling_validation/"
p_width = 8  # inches
p_height = 5.7  # inches
dpi = 100


# photos_lai_in = "C:/Users/Cob/index/educational/usask/research/masters/data/hemispheres/19_149/clean/sized/thresholded/LAI_parsed.dat"
photos_lai_in = "C:/Users/Cob/index/educational/usask/research/masters/data/hemispheres/045_052_050/LAI_045_050_052_parsed.dat"
photos_lai = read.csv(photos_lai_in, header=TRUE, na.strings = c("NA",""), sep=",")
photos_lai$original_file = toupper(gsub("_r.jpg", ".JPG", photos_lai$picture, ignore.case=TRUE))
photos_meta_in = "C:/Users/Cob/index/educational/usask/research/masters/data/hemispheres/hemi_lookup_cleaned.csv"
photos_meta = read.csv(photos_meta_in, header=TRUE, na.strings = c("NA",""), sep=",")
photos_meta$id <- as.numeric(rownames(photos_meta)) - 1

photos = merge(photos_lai, photos_meta, by.x='original_file', by.y='filename', all.x=TRUE)
# photos = photos[, c("original_file", "contactnum_1", "contactnum_2", "contactnum_3", "contactnum_4", "contactnum_5")]
# photos = photos[, c("original_file", "transmission_1", "transmission_2", "transmission_3", "transmission_4", "transmission_5", "transmission_s_1", "transmission_s_2", "transmission_s_3", "transmission_s_4", "transmission_s_5", "contactnum_1", "contactnum_2", "contactnum_3", "contactnum_4", "contactnum_5")]
photos = photos[, c("original_file", "transmission_s_1", "transmission_s_2", "transmission_s_3", "transmission_s_4", "transmission_s_5")]

# rsm_big_in = "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/ray_sampling/batches/lrs_hemi_optimization_r.25_px1000_snow_off/outputs/rshmetalog_footprint_products.csv"
rsm_big_in = "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/ray_sampling/batches/lrs_hemi_optimization_r.25_px1000_snow_on/outputs/rshmetalog_footprint_products.csv"
rsm_big = read.csv(rsm_big_in, header=TRUE, na.strings = c("NA",""), sep=",")
rsm_big$id = as.character(rsm_big$id)
rsm_big = rsm_big[, c("id", "lrs_cn_1", "lrs_cn_2", "lrs_cn_3", "lrs_cn_4", "lrs_cn_5")]
colnames(rsm_big) = c("id", "lrs_cn_big_1", "lrs_cn_big_2", "lrs_cn_big_3", "lrs_cn_big_4", "lrs_cn_big_5")

# rsm_far_in = "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/ray_sampling/batches/lrs_hemi_optimization_r.25_px181_snow_off_max150m/outputs/rshmetalog_footprint_products.csv"
rsm_far_in = "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/ray_sampling/batches/lrs_hemi_optimization_r.25_px181_snow_on_max150m/outputs/rshmetalog_footprint_products.csv"
rsm_far = read.csv(rsm_far_in, header=TRUE, na.strings = c("NA",""), sep=",")
rsm_far$id = as.character(rsm_far$id)
rsm_far = rsm_far[, c("id", "lrs_cn_1", "lrs_cn_2", "lrs_cn_3", "lrs_cn_4", "lrs_cn_5")]
colnames(rsm_far) = c("id", "lrs_cn_far_1", "lrs_cn_far_2", "lrs_cn_far_3", "lrs_cn_far_4", "lrs_cn_far_5")

# rsm_bin_in = "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/ray_sampling/batches/lrs_hemi_optimization_r.25_px181_snow_off/outputs/rshmetalog_footprint_products_threshold.csv"
rsm_bin_in = "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/ray_sampling/batches/lrs_hemi_optimization_r.25_px181_snow_on/outputs/rshmetalog_footprint_products_threshold.csv"
rsm_bin = read.csv(rsm_bin_in, header=TRUE, na.strings = c("NA",""), sep=",")
rsm_bin$id = as.character(rsm_bin$id)
rsm_bin = rsm_bin[, c("id", "lrs_cn_1", "lrs_cn_2", "lrs_cn_3", "lrs_cn_4", "lrs_cn_5", "lrs_tx_1", "lrs_tx_2", "lrs_tx_3", "lrs_tx_4", "lrs_tx_5")]
colnames(rsm_bin) = c("id", "lrs_cn_bin_1", "lrs_cn_bin_2", "lrs_cn_bin_3", "lrs_cn_bin_4", "lrs_cn_bin_5", "lrs_tx_bin_1", "lrs_tx_bin_2", "lrs_tx_bin_3", "lrs_tx_bin_4", "lrs_tx_bin_5")

# rsm_raw_in = "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/ray_sampling/batches/lrs_hemi_optimization_r.25_px181_snow_off/outputs/rshmetalog_footprint_products.csv"
rsm_raw_in = "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/ray_sampling/batches/lrs_hemi_optimization_r.25_px181_snow_on/outputs/rshmetalog_footprint_products.csv"
rsm_raw = read.csv(rsm_raw_in, header=TRUE, na.strings = c("NA",""), sep=",")
rsm_raw$id = as.character(rsm_raw$id)
rsm_raw = rsm_raw[, c("id", "lrs_cn_1", "lrs_cn_2", "lrs_cn_3", "lrs_cn_4", "lrs_cn_5", "lrs_tx_1", "lrs_tx_2", "lrs_tx_3", "lrs_tx_4", "lrs_tx_5")]

# rsm_scl_in = "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/ray_sampling/batches/lrs_hemi_optimization_r.25_px181_snow_off/outputs/rshmetalog_footprint_products_scaled.csv"
rsm_scl_in = "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/ray_sampling/batches/lrs_hemi_optimization_r.25_px181_snow_on/outputs/rshmetalog_footprint_products_scaled.csv"
rsm_scl = read.csv(rsm_scl_in, header=TRUE, na.strings = c("NA",""), sep=",")
rsm_scl$id = as.character(rsm_scl$id)
rsm_scl = rsm_scl[, c("id", "lrs_cn_1", "lrs_cn_2", "lrs_cn_3", "lrs_cn_4", "lrs_cn_5", "lrs_tx_1", "lrs_tx_2", "lrs_tx_3", "lrs_tx_4", "lrs_tx_5")]
colnames(rsm_scl) = c("id", "lrs_cn_scl_1", "lrs_cn_scl_2", "lrs_cn_scl_3", "lrs_cn_scl_4", "lrs_cn_scl_5", "lrs_tx_scl_1", "lrs_tx_scl_2", "lrs_tx_scl_3", "lrs_tx_scl_4", "lrs_tx_scl_5")


rsm_merge = merge(rsm_far, rsm_big, by='id')
rsm_merge = merge(rsm_bin, rsm_merge, by='id')
rsm_merge = merge(rsm_raw, rsm_merge, by='id')
rsm_merge = merge(rsm_scl, rsm_merge, by='id')
rsm_df = rsm_merge %>%
  gather(key, value, -id) %>%
  extract(key, c("val_type", "ring_number"), "(\\D+)_(\\d)") %>%
  spread(val_type, value) %>%
  mutate(angle_mean = (as.numeric(ring_number) * 15 - 15/2) * pi / 180)

ggplot(rsm_df, aes(x=lrs_cn, y=lrs_cn_big, color=ring_number)) +
  geom_abline(intercept = 0, slope = 1) +
  geom_point() +
  labs(title="Mean band-wise expected returns E[<u>] resolution sensitivity (snow-on)", x='E[<u>] (1-degree resolution)', y='E[<u>] (0.18-degree resolution)', color='Zenith angle\nband [deg]') +
  scale_color_discrete(labels = c("0-15", "15-30", "30-45", "45-60", "60-75"), breaks=c(1, 2, 3, 4, 5))
# ggsave(paste0(plot_out_dir, "snow_off_cn_res_eval.png"), width=p_width, height=p_height, dpi=dpi)
# ggsave(paste0(plot_out_dir, "snow_on_cn_res_eval.png"), width=p_width, height=p_height, dpi=dpi)
size_lm = lm(lrs_cn_big ~ lrs_cn, data = rsm_df)
summary(size_lm)

ggplot(rsm_df, aes(x=lrs_cn, y=lrs_cn_far, color=ring_number)) +
  geom_abline(intercept = 0, slope = 1) +
  geom_point() +
  labs(title="Mean band-wise expected returns <u> max distance sensitivity (snow-on)", x='E[<u>] (max 50m)', y='E[<u>] (max 150m)', color='Zenith angle\nband [deg]') +
  scale_color_discrete(labels = c("0-15", "15-30", "30-45", "45-60", "60-75"), breaks=c(1, 2, 3, 4, 5))
# ggsave(paste0(plot_out_dir, "snow_off_cn_max_eval.png"), width=p_width, height=p_height, dpi=dpi)
# ggsave(paste0(plot_out_dir, "snow_on_cn_max_eval.png"), width=p_width, height=p_height, dpi=dpi)
max_lm = lm(lrs_cn_far ~ 0 + lrs_cn, data = rsm_df)
summary(max_lm)

ggplot(rsm_df, aes(x=-log(lrs_tx_scl), y=lrs_cn_scl, color=ring_number)) +
  geom_abline(intercept = 0, slope = 1) +
  geom_point() +
  labs(title="Mean band-wise contact number E(X) commutation response (snow-on)", x='-ln(E[exp(-X)])', y='E[X]', color='Zenith angle\nband [deg]') +
  scale_color_discrete(labels = c("0-15", "15-30", "30-45", "45-60", "60-75"), breaks=c(1, 2, 3, 4, 5))
# ggsave(paste0(plot_out_dir, "snow_off_cn_bin_eval.png"), width=p_width, height=p_height, dpi=dpi)
# ggsave(paste0(plot_out_dir, "snow_on_cn_bin_eval.png"), width=p_width, height=p_height, dpi=dpi)
bin_lm = lm(lrs_cn_far ~ 0 + lrs_cn, data = rsm_df)
summary(bin_lm)

ggplot(rsm_df, aes(x=-log(lrs_tx_bin), y=-log(lrs_tx_scl), color=ring_number)) +
  geom_abline(intercept = 0, slope = 1) +
  geom_point() +
  labs(title="Mean band-wise contact number -ln(E[T]) threshold response (snow-on)", x='-ln(E[H(T - 0.5)])', y='-ln(E[T])', color='Zenith angle\nband [deg]') +
  scale_color_discrete(labels = c("0-15", "15-30", "30-45", "45-60", "60-75"), breaks=c(1, 2, 3, 4, 5))
# ggsave(paste0(plot_out_dir, "snow_off_cn_thresh_eval.png"), width=p_width, height=p_height, dpi=dpi)
# ggsave(paste0(plot_out_dir, "snow_on_cn_thresh_eval.png"), width=p_width, height=p_height, dpi=dpi)
thresh_lm = lm(lrs_cn_far ~ 0 + lrs_cn, data = rsm_df)
summary(thresh_lm)

df = merge(rsm_raw, photos, by.x='id', by.y='original_file')
df = merge(rsm_bin, df, by='id', suffixes = c("_bin", ""))

df = df %>%
  gather(key, value, -id) %>%
  extract(key, c("val_type", "ring_number"), "(\\D+)_(\\d)") %>%
  spread(val_type, value) %>%
  mutate(angle_mean = (as.numeric(ring_number) * 15 - 15/2) * pi / 180)

# calculate error
# ggplot(df, aes(x=lrs_cn, y=-log(transmission_s), color=ring_number)) +
#   geom_point()
# 
# ggplot(df, aes(x=lrs_cn_bin, y=-log(transmission_s), color=ring_number)) +
#   geom_point()


# solid angle weights
df = df %>%
  mutate(solid_angle = 2* pi * (cos((as.numeric(ring_number) - 1) * 15 * pi / 180) - cos(as.numeric(ring_number) * 15 * pi / 180)))

## many models to choose from

# least squares contact number
# lm_rsm_mean_cn = lm(-log(df$transmission_s) ~ 0 + df$lrs_cn)
# summary(lm_rsm_mean_cn)
# cn_lm = predict(lm_rsm_mean_cn, df)
# 
# lmw_rsm_mean_cn = lm(-log(df$transmission_s) ~ 0 + df$lrs_cn, weights=df$solid_angle)
# summary(lmw_rsm_mean_cn)
# cn_lmw = predict(lmw_rsm_mean_cn, df)

# drop ring 5
df_drop = df[df$ring_number != "5",]
lmw_rsm_mean_cn_drop = lm(-log(df_drop$transmission_s) ~ 0 + df_drop$lrs_cn, weights=df_drop$solid_angle)
summary(lmw_rsm_mean_cn_drop)
cn_lmw_drop = predict(lmw_rsm_mean_cn_drop, df_drop)

lmw_rsm_mean_cn_drop_bin = lm(-log(df_drop$transmission_s) ~ 0 + df_drop$lrs_cn_bin, weights=df_drop$solid_angle)
summary(lmw_rsm_mean_cn_drop_bin)
cn_lmw_drop_bin = predict(lmw_rsm_mean_cn_drop_bin, df_drop)

# bin_lm = lm(-log(transmission_s) ~ lrs_cn_bin, data = df)
# summary(bin_lm)

# least squares transmittance

# # equal weight
# nls_rsm_mean_cn = nls(transmission_s ~ exp(-a * lrs_cn), data=df, start=c(a=0.5))
# summary(nls_rsm_mean_cn)
# tx_nls = predict(nls_rsm_mean_cn, newdata=df)

# solid angle weight
# nlsw_rsm_mean_tx = nls(transmission_s ~ exp(-a * lrs_cn), data=df, start=c(a=0.5), weights=solid_angle)
# summary(nlsw_rsm_mean_tx)
# tx_nlsw = predict(nlsw_rsm_mean_tx, newdata=df)
# summary(nlsw_rsm_mean_tx)$parameters[1]
# 
# # solid angle weight, quadratic cn -- both terms are significant... could be good!
# nlsw2_rsm_mean_cn = nls(-log(transmission_s) ~ a * lrs_cn ^ 2 +  b * lrs_cn, data=df, start=c(a=0.5, b=0), weights=solid_angle)
# summary(nlsw2_rsm_mean_cn)
# cn_nlsw2 = predict(nlsw2_rsm_mean_cn, newdata=df)

# # solid angle weight, quadratic tx
# nlsw2_rsm_mean_tx = nls(transmission_s ~ exp(-a * lrs_cn * ( 1 + b * lrs_cn)), data=df, start=c(a=0, b=0.5), weights=solid_angle)
# summary(nlsw2_rsm_mean_tx)
# tx_nlsw2 = predict(nlsw2_rsm_mean_tx, newdata=df)
# 
# test_nls = nls(transmission_s ~ a * exp(-lrs_cn), data = df, start=c(a = 1), weights=solid_angle)
# summary(test_nls)
# tx_test = predict(test_nls, newdata=df)
# 
# wmae_tx = function(cx){
#   tx_error = exp(-cx * df$lrs_cn)  - df$transmission_s
#   weights = df$solid_angle / sum(df$solid_angle, na.rm=TRUE)
#   wmae = abs(sum(weights * tx_error, na.rm=TRUE))
# }
# 
# wmae_cn = function(cx){
#   cn_error = cx * df$lrs_cn  - -log(df$transmission_s)
#   weights = df$solid_angle / sum(df$solid_angle, na.rm=TRUE)
#   wmae = abs(sum(weights * cn_error, na.rm=TRUE))
# }
# 
# opt_tx = optimize(wmae_tx, lower=0, upper=1)
# opt_tx$minimum
# tx_wmae = exp(-opt_tx$minimum * df$lrs_cn)
# 
# opt_cn = optimize(wmae_cn, lower=0, upper=1)
# opt_cn$minimum
# cn_wmae = opt_cn$minimum * df$lrs_cn

# cn vs cn
# ggplot(df_drop, aes(x=lrs_cn)) +
#   geom_point(aes(y=-log(transmission_s), color=ring_number)) +
#   geom_line(aes(y=cn_lmw_drop))
  # geom_line(aes(y=cn_lmw), color="orange") +
  # geom_line(aes(y=-log(tx_test)), color="blue") +
  # geom_line(aes(y=cn_nlsw), color="red") +
  # geom_line(aes(y=-log(tx_nlsw2)), color="brown")
  # geom_line(aes(y=cn_lm), color="red") +
  # geom_line(aes(y=cn_wmae), color="yellow") +
  # geom_line(aes(y=-log(tx_nls)), color="green") +
  # geom_line(aes(y=-log(tx_wmae)), color="purple") +
  # geom_line(aes(y=-log(tx_nlsw2)), color="brown")

# tx vs tx
# ggplot(df, aes(y=transmission_s, x=tx_nlsw2, color=ring_number)) +
#   geom_point() +
#   geom_abline(intercept = 0, slope = 1)

# cn vs tx
# ggplot(df_drop, aes(x=lrs_cn)) +
#   geom_point(aes(y=transmission_s, color=ring_number)) +
#   geom_line(aes(y=tx_test), color="red")
#   geom_line(aes(y=exp(-cn_lm)), color="red") + 
#   geom_line(aes(y=exp(-cn_lmw)), color="orange") +
#   geom_line(aes(y=exp(-cn_wmae)), color="yellow") +
#   geom_line(aes(y=tx_nls), color="green") +
#   geom_line(aes(y=tx_nlsw), color="blue") +
#   geom_line(aes(y=tx_wmae), color="purple") +
#   geom_line(aes(y=tx_nlsw2), color="brown")

# # tx
# ggplot(df_drop, aes(x=exp(-summary(lmw_rsm_mean_cn_drop)$coefficients[1] * lrs_cn), y=transmission_s, color=ring_number)) +
#   geom_point() +
#   geom_abline(intercept = 0, slope = 1) +
#   xlim(0, 1) +
#   ylim(0, 1) +
#   labs(title="Light transmittance (T) error analysis", x='T (ray sampling snow-on)', y='T (hemispherical photography)', color='Zenith angle\nband [deg]') +
#   scale_color_discrete(labels = c("0-15", "15-30", "30-45", "45-60", "60-75"), breaks=c(1, 2, 3, 4, 5))
# # ggsave(paste0(plot_out_dir, "snow_off_tx_error_eval.png"), width=p_width, height=p_height, dpi=dpi)
# # ggsave(paste0(plot_out_dir, "snow_on_tx_error_eval.png"), width=p_width, height=p_height, dpi=dpi)

# cn
ggplot(df_drop, aes(x=summary(lmw_rsm_mean_cn_drop)$coefficients[1] * lrs_cn, y=-log(transmission_s), color=ring_number)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1) +
  labs(title="Mean band-wise contact number E[X] methods comparison (snow-on)", x='E[X] (ray sampling)', y='E[X] (thresholded hemispherical photography)', color='Zenith angle\nband [deg]') +
  scale_color_discrete(labels = c("0-15", "15-30", "30-45", "45-60", "60-75"), breaks=c(1, 2, 3, 4, 5))
# ggsave(paste0(plot_out_dir, "snow_off_cn_photo_error_eval.png"), width=p_width, height=p_height, dpi=dpi)
# ggsave(paste0(plot_out_dir, "snow_on_cn_photo_error_eval.png"), width=p_width, height=p_height, dpi=dpi)

# cn bin
ggplot(df_drop, aes(x=summary(lmw_rsm_mean_cn_drop_bin)$coefficients[1] * lrs_cn_bin, y=-log(transmission_s), color=ring_number)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1) +
  labs(title="Mean band-wise contact number E[X] thresholded methods comparison (snow-on)", x='E[X] (thresholded ray sampling)', y='E[X] (thresholded hemispherical photography)', color='Zenith angle\nband [deg]') +
  scale_color_discrete(labels = c("0-15", "15-30", "30-45", "45-60", "60-75"), breaks=c(1, 2, 3, 4, 5))
# ggsave(paste0(plot_out_dir, "snow_off_cn_bin_photo_error_eval.png"), width=p_width, height=p_height, dpi=dpi)
# ggsave(paste0(plot_out_dir, "snow_on_cn_bin_photo_error_eval.png"), width=p_width, height=p_height, dpi=dpi)



# # # remove 5th ring due to horizon clipping
# df_anal = df[df$ring_number != 5,]
# # lm_rsm_mean_tx = lm(df_anal$contactnum ~ 0 + df_anal$rsm_mean)
# # lm_rsm_mean_cn = lm(-log(df_anal$transmission_s) ~ 0 + df_anal$lrs_cn)
# nls_rsm_mean_cn = nls(transmission_s ~ exp(-a * lrs_cn), data=df_anal, start=c(a=0.5))
# summary(nls_rsm_mean_cn)


fo = paste0("hat(y) == ", sprintf("%.5f",summary(lmw_rsm_mean_cn_drop)$coefficients[1]), " * x")
r2 = paste0("R^2 == ", sprintf("%.5f",summary(lmw_rsm_mean_cn_drop)$adj.r.squared))


ggplot(df_drop, aes(x=lrs_cn, y=-log(transmission_s), color=ring_number)) +
  geom_point() +
  geom_abline(intercept = 0, slope = summary(lmw_rsm_mean_cn_drop)$coefficients[1]) +
  annotate("text", x=5, y=2, label=fo, parse=TRUE) +
  annotate("text", x=5, y=1.9, label=r2, parse=TRUE) +
  labs(title="", x='Lidar returns', y='-log(transmission)', color='Ring')
# ggsave(paste0(plot_out_dir, "snow_off_returns_to_tx_optimization.png"), width=p_width, height=p_height, dpi=dpi)
# ggsave(paste0(plot_out_dir, "snow-on_returns_to_tx_optimization.png"), width=p_width, height=p_height, dpi=dpi)



# stats

model_eval = function(nn = 0){
  
  df_sub = df_drop
  
  if(nn > 0){
    df_sub = df_sub %>%
      filter(ring_number == nn)
  }
  
  # cn model
  tx = df_sub$transmission_s
  cn = -log(tx)
  # lrs_cn = df_sub$lrs_cn * summary(lmw_rsm_mean_cn_drop)$coefficients[1]
  lrs_cn = df_sub$lrs_cn_bin * summary(lmw_rsm_mean_cn_drop_bin)$coefficients[1]
  # lrs_cn = df_sub$lrs_cn * lmw_rsm_mean_cn$coefficients[1]
  # lrs_cn = predict(nlsw2_rsm_mean_cn, newdata=df_sub)
  lrs_tx = exp(-lrs_cn)
  weights = df_sub$solid_angle
  weights = weights / sum(weights)
  
  # # tx model
  # tx = df_sub$transmission_s
  # cn = -log(tx)
  # lrs_tx = predict(nlsw_rsm_mean_tx, newdata=df_sub)
  # lrs_cn = -log(lrs_tx)
  # weights = df_sub$solid_angle
  # weights = weights / sum(weights)
  
  tx_error = lrs_tx - tx
  cn_error = lrs_cn - cn
  
  tx_lm = lm(tx ~ lrs_tx, weights=weights)
  tx_r2_adj = summary(tx_lm)$adj.r.squared
  tx_rmse = summary(tx_lm)$sigma
  tx_p = pf(summary(tx_lm)$fstatistic[1], summary(tx_lm)$fstatistic[2], summary(tx_lm)$fstatistic[3], lower.tail=FALSE)
  
  tx_wrmse = sqrt(sum(weights * (tx_error^2)))
  tx_wmb = sum(tx_error * weights)
  
  cn_lm = lm(cn ~ lrs_cn, weights=weights)
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
# model_eval(5)

# 
# 
# ggplot(df, aes(x=exp(-0.195 * rsm_mean), y=exp(-contactnum), color=ring_number)) +
#   geom_point() +
#   geom_abline(intercept = 0, slope = 1)
# 
# ggplot(df, aes(x=exp(-0.166 * rsm_mean), y=exp(-contactnum), color=ring_number)) +
#   geom_point() +
#   geom_abline(intercept = 0, slope = 1)

##
# 
# # plot linear against nb
# rsm_in = "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/synthetic_hemis/batches/lrs_hemi_optimization_r.25_px361_linear/outputs/contact_number_optimization.csv"
# rsm = read.csv(rsm_in, header=TRUE, na.strings = c("NA",""), sep=",")
# rsm$id = as.character(rsm$id)
# rsm_linear = rsm[, c("id", "rsm_mean_1", "rsm_mean_2", "rsm_mean_3", "rsm_mean_4", "rsm_mean_5", "rsm_med_1", "rsm_med_2", "rsm_med_3", "rsm_med_4", "rsm_med_5")]
# df_l = rsm_linear %>%
#   gather(key, value, -id) %>%
#   extract(key, c("cn_type", "ring_number"), "(\\D+)_(\\d)") %>%
#   spread(cn_type, value)
# 
# rsm_in = "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/synthetic_hemis/batches/lrs_hemi_optimization_r.25_px100_experimental/outputs/contact_number_optimization.csv"
# rsm = read.csv(rsm_in, header=TRUE, na.strings = c("NA",""), sep=",")
# rsm$id = as.character(rsm$id)
# rsm_nb = rsm[, c("id", "rsm_mean_1", "rsm_mean_2", "rsm_mean_3", "rsm_mean_4", "rsm_mean_5", "rsm_med_1", "rsm_med_2", "rsm_med_3", "rsm_med_4", "rsm_med_5")]
# df_nb = rsm_nb %>%
#   gather(key, value, -id) %>%
#   extract(key, c("cn_type", "ring_number"), "(\\D+)_(\\d)") %>%
#   spread(cn_type, value)
# 
# df = merge(df_l, df_nb, by=c('id', 'ring_number'), suffixes = c('_l', '_nb'))
# 
# 
# l_coef = .11992
# nb_coef = .18734
# ggplot(df, aes(x=rsm_mean_l * l_coef, y=rsm_mean_nb * nb_coef, color=ring_number)) +
#   geom_point()
