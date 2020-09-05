library('dplyr')
library('tidyr')
library('ggplot2')

data_in = 'C:/Users/Cob/index/educational/usask/research/masters/data/lidar/analysis/hs_19_045_.25m_canopy_19_149.csv'
data = read.csv(data_in, header=TRUE, na.strings = c("NA",""), sep=",")
data$cc = 1 - data$openness

# plots
# dnt
ggplot(data, aes(x=hs, y=dnt)) +
  geom_point()

# dce
ggplot(data, aes(x=hs, y=dce)) +
  geom_point()

ggplot(data, aes(x=hs, y=dce)) +
  geom_bin2d(binwidth=c(.01, 1))

# chm
ggplot(data, aes(x=hs, y=chm)) +
  geom_point()

ggplot(data, aes(x=hs, y=chm)) +
  geom_bin2d()

# lpmf
ggplot(data, aes(x=hs, y=lpmf)) +
  geom_point()

# lpml
ggplot(data, aes(x=hs, y=lpml)) +
  geom_point()

# lpmc
ggplot(data, aes(x=hs, y=lpmc)) +
  geom_point()

# er_001_mean
ggplot(data, aes(x=hs, y=er_001_mean)) +
  geom_point()

# er_001_median
ggplot(data, aes(x=hs, y=er_001_median)) +
  geom_point() +
  scale_y_log10()

ggplot(data, aes(x=hs, y=er_001_median)) +
  scale_y_log10() +
  geom_bin2d(binwidth=c(.01, .05))

# er_001_sd
ggplot(data, aes(x=hs, y=er_001_sd)) +
  geom_point()

# lai
ggplot(data, aes(x=hs, y=lai_s_cc)) +
  geom_point()

ggplot(data, aes(x=hs, y=lai_s_cc)) +
  geom_bin2d(binwidth=c(.02, .1)) +
  ylim(0, 5)

# cc
ggplot(data, aes(x=hs, y=cc)) +
  geom_point()

# contactnum_1
ggplot(data, aes(x=hs, y=contactnum_1)) +
  geom_point()

# fun zone (only fun beyond this point!)
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
