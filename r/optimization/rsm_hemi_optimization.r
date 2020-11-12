library('dplyr')
library('tidyr')
library('ggplot2')

photos_lai_in = "C:/Users/Cob/index/educational/usask/research/masters/data/hemispheres/19_149/clean/sized/LAI_manual_parsed.dat"
photos_lai = read.csv(photos_lai_in, header=TRUE, na.strings = c("NA",""), sep=",")
photos_lai$original_file = gsub("_r.JPG", ".JPG", photos_lai$picture)
photos_meta_in = "C:/Users/Cob/index/educational/usask/research/masters/data/hemispheres/hemi_lookup_cleaned.csv"
photos_meta = read.csv(photos_meta_in, header=TRUE, na.strings = c("NA",""), sep=",")
photos_meta$id <- as.numeric(rownames(photos_meta)) - 1

photos = merge(photos_lai, photos_meta, by.x='original_file', by.y='filename', all.x=TRUE)
photos = photos[, c("original_file", "contactnum_1", "contactnum_2", "contactnum_3", "contactnum_4", "contactnum_5")]


rsm_in = "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/synthetic_hemis/batches/lrs_hemi_optimization_r.25_px100_beta/outputs/contact_number_optimization.csv"
rsm = read.csv(rsm_in, header=TRUE, na.strings = c("NA",""), sep=",")
rsm$id = as.character(rsm$id)
rsm = rsm[, c("id", "rsm_mean_1", "rsm_mean_2", "rsm_mean_3", "rsm_mean_4", "rsm_mean_5", "rsm_med_1", "rsm_med_2", "rsm_med_3", "rsm_med_4", "rsm_med_5")]

df = merge(rsm, photos, by.x='id', by.y='original_file', all.x=TRUE)


df = df %>%
  gather(key, value, -id) %>%
  extract(key, c("cn_type", "ring_number"), "(\\D+)_(\\d)") %>%
  spread(cn_type, value)
# calculate error

ggplot(df, aes(x=contactnum, y=rsm_mean, color=ring_number)) +
  geom_point()

# remove 5th ring due to horizon clipping

df_anal = df[df$ring_number != 5,]
rsm_mean_lm = lm(df_anal$contactnum ~ 0 + df_anal$rsm_mean)
summary(rsm_mean_lm)


fo = paste0("hat(y) == ", sprintf("%.5f",rsm_mean_lm$coefficients['df_anal$rsm_mean']), " * x")
r2 = paste0("R^2 == ", sprintf("%.5f",summary(rsm_mean_lm)$r.squared))



ggplot(df_anal, aes(x=rsm_mean, y=contactnum, color=ring_number)) +
  geom_point() +
  geom_abline(intercept = 0, slope = rsm_mean_lm$coefficients['df_anal$rsm_mean']) +
  annotate("text", x=5, y=.20, label=fo, parse=TRUE) +
  annotate("text", x=5, y=.15, label=r2, parse=TRUE) +
  labs(title="Contact number from photographic ring analysis vs. expected returns from ray sampling", x='Expected Returns', y='Contact number', color='Ring')

nb_plot
linear_plot

##

# plot linear against nb
rsm_in = "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/synthetic_hemis/batches/lrs_hemi_optimization_r.25_px100_linear/outputs/contact_number_optimization.csv"
rsm = read.csv(rsm_in, header=TRUE, na.strings = c("NA",""), sep=",")
rsm$id = as.character(rsm$id)
rsm_linear = rsm[, c("id", "rsm_mean_1", "rsm_mean_2", "rsm_mean_3", "rsm_mean_4", "rsm_mean_5", "rsm_med_1", "rsm_med_2", "rsm_med_3", "rsm_med_4", "rsm_med_5")]
df_l = rsm_linear %>%
  gather(key, value, -id) %>%
  extract(key, c("cn_type", "ring_number"), "(\\D+)_(\\d)") %>%
  spread(cn_type, value)

rsm_in = "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/synthetic_hemis/batches/lrs_hemi_optimization_r.25_px100_experimental/outputs/contact_number_optimization.csv"
rsm = read.csv(rsm_in, header=TRUE, na.strings = c("NA",""), sep=",")
rsm$id = as.character(rsm$id)
rsm_nb = rsm[, c("id", "rsm_mean_1", "rsm_mean_2", "rsm_mean_3", "rsm_mean_4", "rsm_mean_5", "rsm_med_1", "rsm_med_2", "rsm_med_3", "rsm_med_4", "rsm_med_5")]
df_nb = rsm_nb %>%
  gather(key, value, -id) %>%
  extract(key, c("cn_type", "ring_number"), "(\\D+)_(\\d)") %>%
  spread(cn_type, value)

df = merge(df_l, df_nb, by=c('id', 'ring_number'), suffixes = c('_l', '_nb'))


l_coef = .11992
nb_coef = .18734
ggplot(df, aes(x=rsm_mean_l * l_coef, y=rsm_mean_nb * nb_coef, color=ring_number)) +
  geom_point()
