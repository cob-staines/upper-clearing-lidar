library('dplyr')
library('tidyr')
library('ggplot2')
library('grid')
library('gridExtra')

gd = 'C:/Users/Cob/index/educational/usask/research/masters/graphics/automated/'
plot_h = 21  # cm
plot_w = 29.7  # cm

rr = '.10'

dd = '19_045'
data_in = paste0('C:/Users/Cob/index/educational/usask/research/masters/data/lidar/products/SWE/', dd, '/swe_', dd, '_r', rr, 'm_q0.25_spatial_stats.csv')
data_19_045 = read.csv(data_in, header=TRUE, na.strings = c("NA",""), sep=",")
data_19_045$date = dd

dd = '19_050'
data_in = paste0('C:/Users/Cob/index/educational/usask/research/masters/data/lidar/products/SWE/', dd, '/swe_', dd, '_r', rr, 'm_q0.25_spatial_stats.csv')
data_19_050 = read.csv(data_in, header=TRUE, na.strings = c("NA",""), sep=",")
data_19_050$date = dd

dd = '19_052'
data_in = paste0('C:/Users/Cob/index/educational/usask/research/masters/data/lidar/products/SWE/', dd, '/swe_', dd, '_r', rr, 'm_q0.25_spatial_stats.csv')
data_19_052 = read.csv(data_in, header=TRUE, na.strings = c("NA",""), sep=",")
data_19_052$date = dd

dd = '19_107'
data_in = paste0('C:/Users/Cob/index/educational/usask/research/masters/data/lidar/products/SWE/', dd, '/swe_', dd, '_r', rr, 'm_q0.25_spatial_stats.csv')
data_19_107 = read.csv(data_in, header=TRUE, na.strings = c("NA",""), sep=",")
data_19_107$date = dd

dd = '19_123'
data_in = paste0('C:/Users/Cob/index/educational/usask/research/masters/data/lidar/products/SWE/', dd, '/swe_', dd, '_r', rr, 'm_q0.25_spatial_stats.csv')
data_19_123 = read.csv(data_in, header=TRUE, na.strings = c("NA",""), sep=",")
data_19_123$date = dd

df = rbind(data_19_045, data_19_050, data_19_052, data_19_107, data_19_123)
df$date = as.factor(df$date)



p_swesd = ggplot(data = df, aes(mean_dist, stdev)) +
  facet_grid(. ~ date) +
  geom_point() +
  labs(x='distance (m)', y='standard deviation of SWE (mm)')

p_swe = grid.arrange(p_swesd, nrow=3, top = textGrob('Standard deviation of SWE with distance over the Upper Forest',gp=gpar(fontsize=20,font=3)))
ggsave(paste0(gd, "swe_sd_with_dist.pdf"), p_swe, width = plot_w, height = plot_h, units = "cm")
