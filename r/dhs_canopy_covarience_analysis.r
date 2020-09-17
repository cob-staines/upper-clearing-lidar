library('dplyr')
library('tidyr')
library('ggplot2')
library('grid')
library('gridExtra')

data_in = 'C:/Users/Cob/index/educational/usask/research/masters/data/lidar/analysis/dhs_uf_.25m_canopy_19_149.csv'
gd = 'C:/Users/Cob/index/educational/usask/research/masters/graphics/automated/'
data = read.csv(data_in, header=TRUE, na.strings = c("NA",""), sep=",")
data$cc = 1 - data$openness

# gather snow depths  
data_hs = data %>%
  gather("date", "dhs", c(8, 10, 12, 14, 16))
data_hs$date = as.factor(data_hs$date)
levels(data_hs$date) = c("19_045-19_050", "19_050-19_052", "19_052-19_107", "19_107-19_123", "19_123-19_149")
# drop last date interval
data_hs = data_hs[data_hs$date != "19_123-19_149",]

# reverse sign of dhs
data_hs$dhs = -1 * data_hs$dhs

# plot params
bincount = 50
plot_h = 21  # cm
plot_w = 29.7  # cm
xmin =-.3
xmax = .3
clow = '#005500'
chigh = '#CCFFCC'



######### DISTANCE METRICS

# dnt
p_dnt = ggplot(data_hs, aes(x=dhs, y=dnt)) +
  facet_grid(. ~ date) +
  geom_bin2d(bins = bincount) +
  xlim(xmin, xmax) +
  labs(x='Change in snow depth (m)', y='Distance to nearest tree (m)') +
  scale_fill_gradient(low=clow, high=chigh) +
  theme_minimal()

# dce
p_dce = ggplot(data_hs, aes(x=dhs, y=dce)) +
  facet_grid(. ~ date) +
  geom_bin2d(binwidth=c(.02, .1)) +
  xlim(xmin, xmax) +
  labs(x='Change in snow depth (m)', y='Distance to canopy edge (rectilinear m)') +
  scale_fill_gradient(low=clow, high=chigh) +
  theme_minimal()

# chm
p_chm = ggplot(data_hs, aes(x=dhs, y=chm)) +
  facet_grid(. ~ date) +
  geom_bin2d(bins = bincount) +
  xlim(xmin, xmax) +
  labs(x='Change in snow depth (m)', y='Canopy crown height (m)') +
  scale_fill_gradient(low=clow, high=chigh) +
  theme_minimal()

p_dist = grid.arrange(p_dnt, p_dce, p_chm, nrow=3, top = textGrob("Change in snow depth vs. canopy distance metrics",gp=gpar(fontsize=20,font=3)))
ggsave(paste0(gd, "differential_snow_depth_vs_canopy_distance.pdf"), p_dist, width = plot_w, height = plot_h, units = "cm")

########## LPMs

# lpmf
p_lpmf = ggplot(data_hs, aes(x=dhs, y=lpmf15)) +
  facet_grid(. ~ date) +
  geom_bin2d(bins = bincount) +
  xlim(xmin, xmax) +
  scale_fill_gradient(name = "count", trans = "log", breaks=c(1, 10, 100, 1000), low=clow, high=chigh) +
  labs(x='Change in snow depth (m)', y='LPM first') +
  theme_minimal()

# lpml
p_lpml = ggplot(data_hs, aes(x=dhs, y=lpml15)) +
  facet_grid(. ~ date) +
  geom_bin2d(bins = bincount) +
  xlim(xmin, xmax) +
  scale_fill_gradient(name = "count", trans = "log", breaks=c(1, 10, 100, 1000), low=clow, high=chigh) +
  labs(x='Change in snow depth (m)', y='LPM last') +
  theme_minimal()

# lpmc
p_lpmc = ggplot(data_hs, aes(x=dhs, y=lpmc15)) +
  facet_grid(. ~ date) +
  geom_bin2d(bins = bincount) +
  xlim(xmin, xmax) +
  scale_fill_gradient(name = "count", trans = "log", breaks=c(1, 10, 100, 1000), low=clow, high=chigh) +
  labs(x='Change in snow depth (m)', y='LPM canopy') +
  theme_minimal()

p_lpm = grid.arrange(p_lpmf, p_lpml, p_lpmc, nrow=3, top = textGrob("Change in snow depth vs. laser penetration metrics (LPMs)",gp=gpar(fontsize=20,font=3)))
ggsave(paste0(gd, "differential_snow_depth_vs_lpms.pdf"), p_lpm, width = plot_w, height = plot_h, units = "cm")

########## Expected returns

# er_001_mean
p_erm = ggplot(data_hs, aes(x=dhs, y=er_001_mean)) +
  facet_grid(. ~ date) +
  geom_bin2d(bins = bincount) +
  xlim(xmin, xmax) +
  labs(x='Change in snow depth (m)', y='Mean returns') +
  scale_fill_gradient(low=clow, high=chigh) +
  theme_minimal()

# er_001_median
p_ermed = ggplot(data_hs, aes(x=dhs, y=er_001_median)) +
  facet_grid(. ~ date) +
  geom_bin2d(bins = bincount)+
  xlim(xmin, xmax) +
  labs(x='Change in snow depth (m)', y='Median returns') +
  scale_fill_gradient(low=clow, high=chigh) +
  theme_minimal()

# er_001_sd
p_ersd = ggplot(data_hs, aes(x=dhs, y=er_001_sd)) +
  facet_grid(. ~ date) +
  geom_bin2d(bins = bincount)+
  xlim(xmin, xmax) +
  labs(x='Change in snow depth (m)', y='Standard deviation of returns') +
  scale_fill_gradient(low=clow, high=chigh) +
  theme_minimal()

p_er = grid.arrange(p_erm, p_ermed, p_ersd, nrow=3, top = textGrob("Change in snow depth vs. resampled returns for nadir scans",gp=gpar(fontsize=20,font=3)))
ggsave(paste0(gd, "differential_snow_depth_vs_ray_samples.pdf"), p_er, width = plot_w, height = plot_h, units = "cm")

########## Hemi metrics

# lai
p_lai = ggplot(data_hs, aes(x=dhs, y=lai_s_cc)) +
  facet_grid(. ~ date) +
  geom_bin2d(bins=bincount) +
  xlim(xmin, xmax) +
  ylim(0, 5) +
  labs(x='Change in snow depth (m)', y='LAI') +
  scale_fill_gradient(low=clow, high=chigh) +
  theme_minimal()

# cc
p_cc = ggplot(data_hs, aes(x=dhs, y=cc)) +
  facet_grid(. ~ date) +
  geom_bin2d(bins=bincount) +
  xlim(xmin, xmax) +
  labs(x='Change in snow depth (m)', y='Canopy Closure') +
  scale_fill_gradient(low=clow, high=chigh) +
  theme_minimal()

# contactnum_1
p_cn = ggplot(data_hs, aes(x=dhs, y=contactnum_1)) +
  facet_grid(. ~ date) +
  geom_bin2d(bins=bincount) +
  xlim(xmin, xmax) +
  ylim(0, 2.5) +
  labs(x='Change in snow depth (m)', y='Contact number (0-15\u00B0 from zenith)') +
  scale_fill_gradient(low=clow, high=chigh) +
  theme_minimal()

p_hemi = grid.arrange(p_lai, p_cc, p_cn, nrow=3, top = textGrob("Change in snow depth vs. Hemispherical metrics (Licor-2000 method)",gp=gpar(fontsize=20,font=3)))
ggsave(paste0(gd, "differential_snow_depth_vs_hemi_metrics.pdf"), p_hemi, width = plot_w, height = plot_h, units = "cm")
