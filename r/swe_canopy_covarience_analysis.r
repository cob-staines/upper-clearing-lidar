library('dplyr')
library('tidyr')
library('ggplot2')
library('grid')
library('gridExtra')

data_in = 'C:/Users/Cob/index/educational/usask/research/masters/data/lidar/analysis/swe_uf_.25m_canopy_19_149.csv'
gd = 'C:/Users/Cob/index/educational/usask/research/masters/graphics/automated/'
data = read.csv(data_in, header=TRUE, na.strings = c("NA",""), sep=",")
data$cc = 1 - data$openness

# gather snow deptswe  
data_swe = data %>%
  gather("date", "swe", c(8, 10, 12, 14, 16))
data_swe$date = as.factor(data_swe$date)
levels(data_swe$date) = c("19_045", "19_050", "19_052", "19_107", "19_123")

# plot params
bincount = 50
plot_h = 21  # cm
plot_w = 29.7  # cm
xmin = -50
xmax = 150
clow = '#000055'
chigh = '#CCCCFF'


######### DISTANCE METRICS

# dnt
p_dnt = ggplot(data_swe, aes(x=swe, y=dnt)) +
  facet_grid(. ~ date) +
  geom_bin2d(bins = bincount) +
  xlim(xmin, xmax) +
  labs(x='SWE (mm)', y='Distance to nearest tree (m)') +
  scale_fill_gradient(low=clow, high=chigh) +
  theme_minimal()

# dce
p_dce = ggplot(data_swe, aes(x=swe, y=dce)) +
  facet_grid(. ~ date) +
  geom_bin2d(binwidth=c(3, .1)) +
  xlim(xmin, xmax) +
  labs(x='SWE (mm)', y='Distance to canopy edge (rectilinear m)') +
  scale_fill_gradient(low=clow, high=chigh) +
  theme_minimal()

# chm
p_chm = ggplot(data_swe, aes(x=swe, y=chm)) +
  facet_grid(. ~ date) +
  geom_bin2d(bins = bincount) +
  xlim(xmin, xmax) +
  labs(x='SWE (mm)', y='Canopy crown height (m)') +
  scale_fill_gradient(low=clow, high=chigh) +
  theme_minimal()

p_dist = grid.arrange(p_dnt, p_dce, p_chm, nrow=3, top = textGrob("SWE vs. canopy distance metrics",gp=gpar(fontsize=20,font=3)))
ggsave(paste0(gd, "swe_vs_canopy_distance.pdf"), p_dist, width = plot_w, height = plot_h, units = "cm")

########## LPMs

# lpmf
p_lpmf = ggplot(data_swe, aes(x=swe, y=lpmf15)) +
  facet_grid(. ~ date) +
  geom_bin2d(bins = bincount) +
  xlim(xmin, xmax) +
  scale_fill_gradient(name = "count", trans = "log", breaks=c(1, 10, 100, 1000), low=clow, high=chigh) +
  labs(x='SWE (mm)', y='LPM first') +
  theme_minimal()

# lpml
p_lpml = ggplot(data_swe, aes(x=swe, y=lpml15)) +
  facet_grid(. ~ date) +
  geom_bin2d(bins = bincount) +
  xlim(xmin, xmax) +
  scale_fill_gradient(name = "count", trans = "log", breaks=c(1, 10, 100, 1000), low=clow, high=chigh) +
  labs(x='SWE (mm)', y='LPM last') +
  theme_minimal()

# lpmc
p_lpmc = ggplot(data_swe, aes(x=swe, y=lpmc15)) +
  facet_grid(. ~ date) +
  geom_bin2d(bins = bincount) +
  xlim(xmin, xmax) +
  scale_fill_gradient(name = "count", trans = "log", breaks=c(1, 10, 100, 1000), low=clow, high=chigh) +
  labs(x='SWE (mm)', y='LPM canopy') +
  theme_minimal()

p_lpm = grid.arrange(p_lpmf, p_lpml, p_lpmc, nrow=3, top = textGrob("SWE vs. laser penetration metrics (LPMs)",gp=gpar(fontsize=20,font=3)))
ggsave(paste0(gd, "swe_vs_lpms.pdf"), p_lpm, width = plot_w, height = plot_h, units = "cm")

########## Expected returns

# er_001_mean
p_erm = ggplot(data_swe, aes(x=swe, y=er_001_mean)) +
  facet_grid(. ~ date) +
  geom_bin2d(bins = bincount) +
  xlim(xmin, xmax) +
  labs(x='SWE (mm)', y='Mean returns') +
  scale_fill_gradient(low=clow, high=chigh) +
  theme_minimal()

# er_001_median
p_ermed = ggplot(data_swe, aes(x=swe, y=er_001_median)) +
  facet_grid(. ~ date) +
  geom_bin2d(bins = bincount)+
  xlim(xmin, xmax) +
  labs(x='SWE (mm)', y='Median returns') +
  scale_fill_gradient(low=clow, high=chigh) +
  theme_minimal()

# er_001_sd
p_ersd = ggplot(data_swe, aes(x=swe, y=er_001_sd)) +
  facet_grid(. ~ date) +
  geom_bin2d(bins = bincount)+
  xlim(xmin, xmax) +
  labs(x='SWE (mm)', y='Standard deviation of returns') +
  scale_fill_gradient(low=clow, high=chigh) +
  theme_minimal()

p_er = grid.arrange(p_erm, p_ermed, p_ersd, nrow=3, top = textGrob("SWE vs. resampled returns for nadir scans",gp=gpar(fontsize=20,font=3)))
ggsave(paste0(gd, "swe_vs_ray_samples.pdf"), p_er, width = plot_w, height = plot_h, units = "cm")

########## Hemi metrics

# lai
p_lai = ggplot(data_swe, aes(x=swe, y=lai_s_cc)) +
  facet_grid(. ~ date) +
  geom_bin2d(bins=bincount) +
  xlim(xmin, xmax) +
  ylim(0, 5) +
  labs(x='SWE (mm)', y='LAI') +
  scale_fill_gradient(low=clow, high=chigh) +
  theme_minimal()

# cc
p_cc = ggplot(data_swe, aes(x=swe, y=cc)) +
  facet_grid(. ~ date) +
  geom_bin2d(bins=bincount) +
  xlim(xmin, xmax) +
  labs(x='SWE (mm)', y='Canopy Closure') +
  scale_fill_gradient(low=clow, high=chigh) +
  theme_minimal()

# contactnum_1
p_cn = ggplot(data_swe, aes(x=swe, y=contactnum_1)) +
  facet_grid(. ~ date) +
  geom_bin2d(bins=bincount) +
  xlim(xmin, xmax) +
  ylim(0, 2.5) +
  labs(x='SWE (mm)', y='Contact number (0-15\u00B0 from zenith)') +
  scale_fill_gradient(low=clow, high=chigh) +
  theme_minimal()

p_hemi = grid.arrange(p_lai, p_cc, p_cn, nrow=3, top = textGrob("SWE vs. Hemispherical metrics (Licor-2000 method)",gp=gpar(fontsize=20,font=3)))
ggsave(paste0(gd, "swe_vs_hemi_metrics.pdf"), p_hemi, width = plot_w, height = plot_h, units = "cm")


####################
# fun zone (only fun beyond this point!)

# snow depth vs snow depth for different days
ggplot(data, aes(x=swe_19_045, y=swe_19_052)) +
  geom_bin2d(bins = bincount) +
  geom_abline(slope=1, intercept=0) +
  theme_minimal()

ggplot(data, aes(x=hs_19_045, y=count_19_045 + count_19_149)) +
  geom_point() +
  theme_minimal() +
  scale_y_log10()


ggplot(data, aes(x=hs_19_045, y=hs_19_052)) +
  geom_bin2d(bins = bincount) +
  geom_abline(slope=1, intercept=0) +
  theme_minimal()


data$low_counts = 'no'
data$low_counts[data$count_19_045 < 10] = '19_045'
data$low_counts[data$count_19_050 < 10] = '19_050'
data$low_counts[data$count_19_149 < 10] = '19_149'
data$low_counts = as.factor(data$low_counts)
data %>%
  filter(low_counts != 'no') %>%
  ggplot(., aes(x=hs_19_045, y=hs_19_052, color=low_counts)) +
    geom_point() +
    geom_abline(slope=1, intercept=0) +
    theme_minimal() +
    scale_fill_gradient(name = "count", trans = "log", breaks=c(1, 10, 100), low=clow, high=chigh)

# point counts vs snow depth... this should tell you something about bias.
ggplot(data, aes(x=swe_19_045, y=count_19_045)) +
  geom_bin2d(bins = bincount) +
  ylim(0, 250) +
  theme_minimal()

# comparison of er_median densities over full forest and over points with snow depth (how representative are our snow dempth samples?)
df2 = data[(data$count_19_045 > 0) & (data$count_19_149 > 0),]
ggplot(data, aes(x=er_001_median)) +
  geom_density() +
  geom_density(data = df2, color = "red")

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
