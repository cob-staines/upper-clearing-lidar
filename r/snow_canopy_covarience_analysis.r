library('dplyr')
library('tidyr')
library('ggplot2')
library('grid')
library('gridExtra')

data_in = 'C:/Users/Cob/index/educational/usask/research/masters/data/lidar/analysis/hs_uf_.25m_canopy_19_149.csv'
gd = 'C:/Users/Cob/index/educational/usask/research/masters/graphics/automated/'
data = read.csv(data_in, header=TRUE, na.strings = c("NA",""), sep=",")
data$cc = 1 - data$openness

# gather snow depths  
data_hs = data %>%
  gather("date", "hs", c(8, 10, 12, 14, 16))
data_hs$date = as.factor(data_hs$date)
levels(data_hs$date) = c("19_045", "19_050", "19_052", "19_107", "19_123")

# plot params
bincount = 50
plot_h = 21  # cm
plot_w = 29.7  # cm


# snow depth vs snow depth for different days
ggplot(data, aes(x=hs_19_045, y=hs_19_123)) +
  geom_bin2d(bins = bincount) +
  geom_abline(slope=1, intercept=0) +
  theme_minimal()

# point counts vs snow depth... this should tell you something about bias.
ggplot(data, aes(x=hs_19_045, y=count_19_045)) +
  geom_bin2d(bins = bincount) +
  ylim(0, 250) +
  theme_minimal()

# comparison of er_median densities over full forest and over points with snow depth (how representative are our snow dempth samples?)
df2 = data[(data$count_19_045 > 0) & (data$count_19_149 > 0),]
ggplot(data, aes(x=er_001_median)) +
  geom_density() +
  geom_density(data = df2, color = "red")

######### DISTANCE METRICS

# dnt
p_dnt = ggplot(data_hs, aes(x=hs, y=dnt)) +
  facet_grid(. ~ date) +
  geom_bin2d(bins = bincount) +
  labs(x='Snow depth (m)', y='Distance to nearest tree (m)') +
  theme_minimal()

# dce
p_dce = ggplot(data_hs, aes(x=hs, y=dce)) +
  facet_grid(. ~ date) +
  geom_bin2d(binwidth=c(.02, .1)) +
  labs(x='Snow depth (m)', y='Distance to canopy edge (rectilinear m)') +
  theme_minimal()

# chm
p_chm = ggplot(data_hs, aes(x=hs, y=chm)) +
  facet_grid(. ~ date) +
  geom_bin2d(bins = bincount) +
  labs(x='Snow depth (m)', y='Canopy crown height (m)') +
  theme_minimal()

p_dist = grid.arrange(p_dnt, p_dce, p_chm, nrow=3, top = textGrob("Snow depth vs. canopy distance metrics",gp=gpar(fontsize=20,font=3)))
ggsave(paste0(gd, "snow_depth_vs_canopy_distance.pdf"), p_dist, width = plot_w, height = plot_h, units = "cm")

########## LPMs

# lpmf
p_lpmf = ggplot(data_hs, aes(x=hs, y=lpmf)) +
  facet_grid(. ~ date) +
  geom_bin2d(bins = bincount) +
  scale_fill_gradient(name = "count", trans = "log", breaks=c(1, 10, 100, 1000)) +
  labs(x='Snow depth (m)', y='LPM first') +
  theme_minimal()

# lpml
p_lpml = ggplot(data_hs, aes(x=hs, y=lpml)) +
  facet_grid(. ~ date) +
  geom_bin2d(bins = bincount) +
  scale_fill_gradient(name = "count", trans = "log", breaks=c(1, 10, 100, 1000)) +
  labs(x='Snow depth (m)', y='LPM last') +
  theme_minimal()

# lpmc
p_lpmc = ggplot(data_hs, aes(x=hs, y=lpmc)) +
  facet_grid(. ~ date) +
  geom_bin2d(bins = bincount) +
  scale_fill_gradient(name = "count", trans = "log", breaks=c(1, 10, 100, 1000)) +
  labs(x='Snow depth (m)', y='LPM canopy') +
  theme_minimal()

p_lpm = grid.arrange(p_lpmf, p_lpml, p_lpmc, nrow=3, top = textGrob("Snow depth vs. laser penetration metrics (LPMs)",gp=gpar(fontsize=20,font=3)))
ggsave(paste0(gd, "snow_depth_vs_lpms.pdf"), p_lpm, width = plot_w, height = plot_h, units = "cm")

########## Expected returns

# er_001_mean
p_erm = ggplot(data_hs, aes(x=hs, y=er_001_mean)) +
  facet_grid(. ~ date) +
  geom_bin2d(bins = bincount) +
  labs(x='Snow depth (m)', y='Mean returns') +
  theme_minimal()

# er_001_median
p_ermed = ggplot(data_hs, aes(x=hs, y=er_001_median)) +
  facet_grid(. ~ date) +
  geom_bin2d(bins = bincount)+
  labs(x='Snow depth (m)', y='Median returns') +
  theme_minimal()

# er_001_sd
p_ersd = ggplot(data_hs, aes(x=hs, y=er_001_sd)) +
  facet_grid(. ~ date) +
  geom_bin2d(bins = bincount)+
  labs(x='Snow depth (m)', y='Standard deviation of returns') +
  theme_minimal()

p_er = grid.arrange(p_erm, p_ermed, p_ersd, nrow=3, top = textGrob("Snow depth vs. resampled returns for nadir scans",gp=gpar(fontsize=20,font=3)))
ggsave(paste0(gd, "snow_depth_vs_ray_samples.pdf"), p_er, width = plot_w, height = plot_h, units = "cm")

########## Hemi metrics

# lai
p_lai = ggplot(data_hs, aes(x=hs, y=lai_s_cc)) +
  facet_grid(. ~ date) +
  geom_bin2d(bins=bincount) +
  ylim(0, 5) +
  labs(x='Snow depth (m)', y='LAI') +
  theme_minimal()

# cc
p_cc = ggplot(data_hs, aes(x=hs, y=cc)) +
  facet_grid(. ~ date) +
  geom_bin2d(bins=bincount) +
  labs(x='Snow depth (m)', y='Canopy Closure') +
  theme_minimal()

# contactnum_1
p_cn = ggplot(data_hs, aes(x=hs, y=contactnum_1)) +
  facet_grid(. ~ date) +
  geom_bin2d(bins=bincount) +
  ylim(0, 2.5) +
  labs(x='Snow depth (m)', y='Contact number (0-15 degrees from zenith)') +
  theme_minimal()

p_hemi = grid.arrange(p_lai, p_cc, p_cn, nrow=3, top = textGrob("Snow depth vs. Hemispherical metrics (Licor-2000 method)",gp=gpar(fontsize=20,font=3)))
ggsave(paste0(gd, "snow_depth_vs_hemi_metrics.pdf"), p_hemi, width = plot_w, height = plot_h, units = "cm")


# fun zone (only fun beyond this point!)
ggplot(data, aes(x=er_001_median, y=count)) +
  geom_bin2d()

ggplot(data, aes(x=chm, y=er_001_median)) +
  geom_point()

ggplot(data, aes(x=lpmf, y=er_001_median)) +
  geom_point()

ggplot(data, aes(x=lpml, y=er_001_median)) +
  geom_point()

ggplot(data, aes(x=lpmc, y=er_001_median)) +
  geom_point()

ggplot(data, aes(lai_s_cc, cc)) +
  geom_point()

ggplot(data, aes(dnt, dce)) +
  geom_point()

ggplot(data, aes(x=lai_no_cor, y=lai_s_cc)) +
  geom_point() +
  geom_abline(slope=1, intercept=0)

ggplot(data, aes(x=contactnum_1, y=er_001_median)) +
  geom_bin2d(binwidth=c(.1, 1))
# why are contact num and expected returns seemingly inversely related? I expected at least a blurry positive signal...

ggplot(data, aes(x=chm, y=er_001_median)) +
  geom_point()
