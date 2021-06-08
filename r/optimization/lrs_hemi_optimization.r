library('dplyr')
library('tidyr')
library('ggplot2')

plot_out_dir = "C:/Users/Cob/index/educational/usask/research/masters/graphics/thesis_graphics/validation/ray_sampling_validation/"
p_width = 8  # inches
p_height = 5.7  # inches
dpi = 100


photos_lai_in = "C:/Users/Cob/index/educational/usask/research/masters/data/hemispheres/19_149/clean/sized/thresholded/LAI_parsed.dat"
# photos_lai_in = "C:/Users/Cob/index/educational/usask/research/masters/data/hemispheres/045_052_050/LAI_045_050_052_parsed.dat"
photos_lai = read.csv(photos_lai_in, header=TRUE, na.strings = c("NA",""), sep=",")
photos_lai$original_file = toupper(gsub("_r.jpg", ".JPG", photos_lai$picture, ignore.case=TRUE))
photos_meta_in = "C:/Users/Cob/index/educational/usask/research/masters/data/hemispheres/hemi_lookup_cleaned.csv"
photos_meta = read.csv(photos_meta_in, header=TRUE, na.strings = c("NA",""), sep=",")
photos_meta$id <- as.numeric(rownames(photos_meta)) - 1

photos = merge(photos_lai, photos_meta, by.x='original_file', by.y='filename', all.x=TRUE)
# photos = photos[, c("original_file", "contactnum_1", "contactnum_2", "contactnum_3", "contactnum_4", "contactnum_5")]
# photos = photos[, c("original_file", "transmission_1", "transmission_2", "transmission_3", "transmission_4", "transmission_5", "transmission_s_1", "transmission_s_2", "transmission_s_3", "transmission_s_4", "transmission_s_5", "contactnum_1", "contactnum_2", "contactnum_3", "contactnum_4", "contactnum_5")]
photos = photos[, c("original_file", "transmission_s_1", "transmission_s_2", "transmission_s_3", "transmission_s_4", "transmission_s_5")]


# rsm_in = "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/ray_sampling/batches/lrs_hemi_optimization_r.25_px181_beta_single_ray_agg_045_050_052/outputs/contact_number_optimization.csv"
# rsm = read.csv(rsm_in, header=TRUE, na.strings = c("NA",""), sep=",")
# rsm$id = as.character(rsm$id)
# rsm = rsm[, c("id", "rsm_mean_1", "rsm_mean_2", "rsm_mean_3", "rsm_mean_4", "rsm_mean_5", "rsm_std_1", "rsm_std_2", "rsm_std_3", "rsm_std_4", "rsm_std_5")]

rsm_2_in = "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/ray_sampling/batches/lrs_hemi_optimization_r.25_px181_snow_off/outputs/rshmetalog_footprint_products.csv"
# rsm_2_in = "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/ray_sampling/batches/lrs_hemi_optimization_r.25_px181_snow_on/outputs/rshmetalog_footprint_products.csv"
rsm_2 = read.csv(rsm_2_in, header=TRUE, na.strings = c("NA",""), sep=",")
rsm_2$id = as.character(rsm_2$id)
rsm_2 = rsm_2[, c("id", "lrs_cn_1", "lrs_cn_2", "lrs_cn_3", "lrs_cn_4", "lrs_cn_5")]


df = merge(rsm_2, photos, by.x='id', by.y='original_file', all.x=TRUE)


df = df %>%
  gather(key, value, -id) %>%
  extract(key, c("val_type", "ring_number"), "(\\D+)_(\\d)") %>%
  spread(val_type, value) %>%
  mutate(angle_mean = (as.numeric(ring_number) * 15 - 15/2) * pi / 180)

# calculate error
ggplot(df, aes(x=-log(transmission_s), y=lrs_cn, color=ring_number)) +
  geom_point()


# solid angle weights
df = df %>%
  # filter(id != "19_149_DSCN6393.JPG" & id != "19_149_DSCN6395.JPG") %>%  # drop first 2 images?
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

# least squares transmittance

# # equal weight
# nls_rsm_mean_cn = nls(transmission_s ~ exp(-a * lrs_cn), data=df, start=c(a=0.5))
# summary(nls_rsm_mean_cn)
# tx_nls = predict(nls_rsm_mean_cn, newdata=df)

# solid angle weight
nlsw_rsm_mean_cn = nls(transmission_s ~ exp(-a * lrs_cn), data=df, start=c(a=0.5), weights=solid_angle)
summary(nlsw_rsm_mean_cn)
tx_nlsw = predict(nlsw_rsm_mean_cn, newdata=df)
summary(nlsw_rsm_mean_cn)$parameters[1]

# # solid angle weight, quadratic
# nlsw2_rsm_mean_cn = nls(transmission_s ~ exp(-a * lrs_cn * ( 1 + b * lrs_cn)), data=df, start=c(a=0.5, b=0), weights=solid_angle)
# summary(nlsw2_rsm_mean_cn)
# tx_nlsw2 = predict(nlsw2_rsm_mean_cn, newdata=df)

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

# # cn vs cn
# ggplot(df, aes(x=lrs_cn)) +
#   geom_point(aes(y=-log(transmission_s), color=ring_number)) +
#   geom_line(aes(y=cn_lm), color="red") + 
#   geom_line(aes(y=cn_lmw), color="orange") +
#   geom_line(aes(y=cn_wmae), color="yellow") +
#   geom_line(aes(y=-log(tx_nls)), color="green") +
#   geom_line(aes(y=-log(tx_nlsw)), color="blue") +
#   geom_line(aes(y=-log(tx_wmae)), color="purple") +
#   geom_line(aes(y=-log(tx_nlsw2)), color="brown")
# 
# # cn vs tx
# ggplot(df, aes(x=lrs_cn)) +
#   geom_point(aes(y=transmission_s, color=ring_number)) +
#   geom_line(aes(y=exp(-cn_lm)), color="red") + 
#   geom_line(aes(y=exp(-cn_lmw)), color="orange") +
#   geom_line(aes(y=exp(-cn_wmae)), color="yellow") +
#   geom_line(aes(y=tx_nls), color="green") +
#   geom_line(aes(y=tx_nlsw), color="blue") +
#   geom_line(aes(y=tx_wmae), color="purple") +
#   geom_line(aes(y=tx_nlsw2), color="brown")

ggplot(df, aes(x=exp(-0.1842 * lrs_cn), y=transmission_s, color=ring_number)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1) +
  xlim(0, 1) +
  ylim(0, 1) +
  labs(title="Light transmittance (T) error analysis", x='T (ray sampling snow-on)', y='T (hemispherical photography)', color='Zenith angle\nband [deg]') +
  scale_color_discrete(labels = c("0-15", "15-30", "30-45", "45-60", "60-75"), breaks=c(1, 2, 3, 4, 5))
# ggsave(paste0(plot_out_dir, "snow_off_tx_error_eval.png"), width=p_width, height=p_height, dpi=dpi)
# ggsave(paste0(plot_out_dir, "snow_on_tx_error_eval.png"), width=p_width, height=p_height, dpi=dpi)
  

# # # remove 5th ring due to horizon clipping
# df_anal = df[df$ring_number != 5,]
# # lm_rsm_mean_tx = lm(df_anal$contactnum ~ 0 + df_anal$rsm_mean)
# # lm_rsm_mean_cn = lm(-log(df_anal$transmission_s) ~ 0 + df_anal$lrs_cn)
# nls_rsm_mean_cn = nls(transmission_s ~ exp(-a * lrs_cn), data=df_anal, start=c(a=0.5))
# summary(nls_rsm_mean_cn)


fo = paste0("hat(y) == ", sprintf("%.5f",lm_rsm_mean_cn$coefficients['df$lrs_cn']), " * x")
r2 = paste0("R^2 == ", sprintf("%.5f",summary(lm_rsm_mean_cn)$adj.r.squared))


ggplot(df, aes(x=lrs_cn, y=-log(transmission_s), color=ring_number)) +
  geom_point() +
  geom_abline(intercept = 0, slope = lm_rsm_mean_cn$coefficients['df$lrs_cn']) +
  annotate("text", x=5, y=4, label=fo, parse=TRUE) +
  annotate("text", x=5, y=3.8, label=r2, parse=TRUE) +
  labs(title="", x='Lidar returns', y='-log(transmission)', color='Ring')
ggsave(paste0(plot_out_dir, "19_149_returns_to_tx_optimization.png"), width=p_width, height=p_height, dpi=dpi)
# ggsave(paste0(plot_out_dir, "045_050_052_returns_to_tx_optimization.png"), width=p_width, height=p_height, dpi=dpi)


ggplot(df, aes(x=exp(-lm_rsm_mean_cn$coefficients['df$lrs_cn'] * lrs_cn), y=transmission_s, color=ring_number)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1) +
  labs(title="", x='Voxel transmission', y='Hemiphoto transmission', color='Ring')

# lm_rsm_mean_tx = lm(df$transmission_s ~ 0 + exp(-lm_rsm_mean_cn$coefficients['df$lrs_cn'] * df$lrs_cn))
# summary(lm_rsm_mean_tx)

tx_ssres = sum((tx_nls - df$transmission_s)^2, na.rm=TRUE)
tx_sstot = sum((mean(df$transmission_s, na.rm=TRUE) - df$transmission_s)^2, na.rm=TRUE)
tx_r2 = 1 - tx_ssres / tx_sstot
tx_r2_adj = 1 - (1 - tx_r2) * (sum(!is.na(df$transmission_s)) - 1) / (sum(!is.na(df$transmission_s)) - 2)

cn_ssres = sum((-log(tx_nls) - -log(df$transmission_s))^2, na.rm=TRUE)
cn_sstot = sum((mean(-log(df$transmission_s), na.rm=TRUE) - -log(df$transmission_s))^2, na.rm=TRUE)
cn_r2 = 1 - cn_ssres / cn_sstot
cn_r2_adj = 1 - (1 - cn_r2) * (sum(!is.na(-log(df$transmission_s))) - 1) / (sum(!is.na(-log(df$transmission_s))) - 2)

cor(tx_nls, df$transmission_s) ^2
cor(-log(tx_nls), -log(df$transmission_s)) ^2

x = -log(tx_nls)
y = -log(df$transmission_s)
summary(lm(y ~ x))

x = tx_nls
y = df$transmission_s
summary(lm(y ~ x))

# stats
rmse_cn = sqrt(mean((lm_rsm_mean_cn$coefficients['df$lrs_cn'] * df$lrs_cn - -log(df$transmission_s))^2, na.rm=TRUE))
rmse_tx = sqrt(mean((exp(-lm_rsm_mean_cn$coefficients['df$lrs_cn'] * df$lrs_cn) - df$transmission_s)^2, na.rm=TRUE))

model_eval = function(nn = 0){
  
  df_sub = df
  
  if(nn > 0){
    df_sub = df_sub %>%
      filter(ring_number == nn)
  }
  
  tx = df_sub$transmission_s
  cn = -log(tx)
  lrs_tx = predict(nlsw_rsm_mean_cn, newdata=df_sub)
  lrs_cn = -log(lrs_tx)
  weights = df_sub$solid_angle
  
  tx_error = lrs_tx - tx
  cn_error = lrs_cn - cn
  
  tx_lm = lm(tx ~ lrs_tx, weights=weights)
  tx_r2_adj = summary(tx_lm)$adj.r.squared
  tx_rmse = summary(tx_lm)$sigma
  
  cn_lm = lm(cn ~ lrs_cn, weights=weights)
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
