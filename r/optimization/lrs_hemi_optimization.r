library('dplyr')
library('tidyr')
library('ggplot2')

plot_out_dir = "C:/Users/Cob/index/educational/usask/research/masters/graphics/thesis_graphics/validation/ray_sampling_validation/"
p_width = 8  # inches
p_height = 5.7  # inches
dpi = 100


# photos_lai_in = "C:/Users/Cob/index/educational/usask/research/masters/data/hemispheres/19_149/clean/sized/LAI_manual_parsed.dat"
photos_lai_in = "C:/Users/Cob/index/educational/usask/research/masters/data/hemispheres/045_052_050/LAI_045_050_052_parsed.dat"
photos_lai = read.csv(photos_lai_in, header=TRUE, na.strings = c("NA",""), sep=",")
photos_lai$original_file = toupper(gsub("_r.JPG", ".JPG", photos_lai$picture))
photos_meta_in = "C:/Users/Cob/index/educational/usask/research/masters/data/hemispheres/hemi_lookup_cleaned.csv"
photos_meta = read.csv(photos_meta_in, header=TRUE, na.strings = c("NA",""), sep=",")
photos_meta$id <- as.numeric(rownames(photos_meta)) - 1

photos = merge(photos_lai, photos_meta, by.x='original_file', by.y='filename', all.x=TRUE)
# photos = photos[, c("original_file", "contactnum_1", "contactnum_2", "contactnum_3", "contactnum_4", "contactnum_5")]
# photos = photos[, c("original_file", "transmission_1", "transmission_2", "transmission_3", "transmission_4", "transmission_5")]
photos = photos[, c("original_file", "transmission_s_1", "transmission_s_2", "transmission_s_3", "transmission_s_4", "transmission_s_5")]


# rsm_in = "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/ray_sampling/batches/lrs_hemi_optimization_r.25_px181_beta_single_ray_agg_045_050_052/outputs/contact_number_optimization.csv"
# rsm = read.csv(rsm_in, header=TRUE, na.strings = c("NA",""), sep=",")
# rsm$id = as.character(rsm$id)
# rsm = rsm[, c("id", "rsm_mean_1", "rsm_mean_2", "rsm_mean_3", "rsm_mean_4", "rsm_mean_5", "rsm_std_1", "rsm_std_2", "rsm_std_3", "rsm_std_4", "rsm_std_5")]

# rsm_2_in = "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/ray_sampling/batches/lrs_hemi_optimization_r.25_px181_beta_single_ray_agg_19_149/outputs/rshmetalog_products.csv"

rsm_2_in = "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/ray_sampling/batches/lrs_hemi_optimization_r.25_px181_beta_single_ray_agg_045_050_052/outputs/rshmetalog_products.csv"
rsm_2 = read.csv(rsm_2_in, header=TRUE, na.strings = c("NA",""), sep=",")
rsm_2$id = as.character(rsm_2$id)
rsm_2 = rsm_2[, c("id", "cn_1", "cn_2", "cn_3", "cn_4", "cn_5")]


df = merge(rsm_2, photos, by.x='id', by.y='original_file', all.x=TRUE)


df = df %>%
  gather(key, value, -id) %>%
  extract(key, c("val_type", "ring_number"), "(\\D+)_(\\d)") %>%
  spread(val_type, value)
# calculate error

# ggplot(df, aes(x=contactnum, y=rsm_mean, color=ring_number)) +
#   geom_point()

# ggplot(df, aes(x=transmission, y=rsm_mean, color=ring_number)) +
#   geom_point()

# ggplot(df, aes(x=-log(transmission), y=rsm_mean, color=ring_number)) +
#   geom_point()

ggplot(df, aes(x=-log(transmission_s), y=cn, color=ring_number)) +
  geom_point()

# rsm_mean_lm_all = lm(df$contactnum ~ 0 + df$rsm_mean)
# summary(rsm_mean_lm_all)

# lm_rsm_mean_tx = lm(-log(df$transmission) ~ 0 + df$rsm_mean)
# summary(lm_rsm_mean_tx)

lm_rsm_mean_tx = lm(-log(df$transmission_s) ~ 0 + df$cn)
summary(lm_rsm_mean_tx)

# remove 5th ring due to horizon clipping
# df_anal = df[df$ring_number != 5,]
# lm_rsm_mean_tx = lm(df_anal$contactnum ~ 0 + df_anal$rsm_mean)
# lm_rsm_mean_tx = lm(-log(df_anal$transmission) ~ 0 + df_anal$rsm_mean)
# summary(lm_rsm_mean_tx)


fo = paste0("hat(y) == ", sprintf("%.5f",lm_rsm_mean_tx$coefficients['df$cn']), " * x")
r2 = paste0("R^2 == ", sprintf("%.5f",summary(lm_rsm_mean_tx)$r.squared))


ggplot(df, aes(x=cn, y=-log(transmission_s), color=ring_number)) +
  geom_point() +
  geom_abline(intercept = 0, slope = lm_rsm_mean_tx$coefficients['df$cn']) +
  annotate("text", x=5, y=4, label=fo, parse=TRUE) +
  annotate("text", x=5, y=3.8, label=r2, parse=TRUE) +
  labs(title="", x='Lidar returns', y='-log(transmission)', color='Ring')
ggsave(paste0(plot_out_dir, "19_149_returns_to_tx_optimization.png"), width=p_width, height=p_height, dpi=dpi)
# ggsave(paste0(plot_out_dir, "045_050_052_returns_to_tx_optimization.png"), width=p_width, height=p_height, dpi=dpi)

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
