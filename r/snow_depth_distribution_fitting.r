library('dplyr')
library('tidyr')
library('ggplot2')
library('grid')
library('gridExtra')

data_in = 'C:/Users/Cob/index/educational/usask/research/masters/data/lidar/analysis/hs_uf_.25m_canopy_19_149.csv'
data = read.csv(data_in, header=TRUE, na.strings = c("NA",""), sep=",")

gd = 'C:/Users/Cob/index/educational/usask/research/masters/graphics/automated/'
plot_h = 21  # cm
plot_w = 29.7  # cm

# gather snow depths  
data_hs = data %>%
  gather("date", "hs", c(8, 10, 12, 14, 16))
data_hs$date = as.factor(data_hs$date)
levels(data_hs$date) = c("19_045", "19_050", "19_052", "19_107", "19_123")



datelist = levels(data_hs$date) = c("19_045", "19_050", "19_052", "19_107", "19_123")
mu <- numeric(length = length(datelist))
sig <- numeric(length = length(datelist))
data_hs$lnsamps = 0
for (ii in 1:length(datelist)){
  date = datelist[ii]
  samples = data_hs$hs[data_hs$date == date]
  mu_x = mean(samples, na.rm=TRUE)
  sig_x = sd(samples, na.rm=TRUE)
  
  mu[ii] = log(mu_x^2/sqrt(mu_x^2 + sig_x^2))
  sig[ii] = sqrt(log(1 + sig_x^2/mu_x^2))
  
  n = length(samples)
  data_hs$lnsamps[data_hs$date == date] = rlnorm(n, meanlog=mu[ii], sdlog=sig[ii])
}


p_045 = ggplot(data_hs, aes(sample=hs)) +
  stat_qq(distribution=stats::qlnorm, na.rm=TRUE, dparams=c(mu[1], sig[1])) +
  geom_abline(intercept=0, slope=1) +
  xlim(0, 1) + 
  labs(title=datelist[1],  x=sprintf(mu[1], sig[1], fmt='theoretical log normal\n(mu=%#.3f, sig=%#.3f)'), y = 'snow depth distribution') +
  theme_minimal()

p_050 = ggplot(data_hs, aes(sample=hs)) +
  stat_qq(distribution=stats::qlnorm, na.rm=TRUE, dparams=c(mu[2], sig[2])) +
  geom_abline(intercept=0, slope=1) +
  xlim(0, 1) + 
  labs(title=datelist[2],  x=sprintf(mu[2], sig[2], fmt='theoretical log normal\n(mu=%#.3f, sig=%#.3f)'), y = 'snow depth distribution') +
  theme_minimal()

p_052 = ggplot(data_hs, aes(sample=hs)) +
  stat_qq(distribution=stats::qlnorm, na.rm=TRUE, dparams=c(mu[3], sig[3])) +
  geom_abline(intercept=0, slope=1) +
  xlim(0, 1) + 
  labs(title=datelist[3],  x=sprintf(mu[3], sig[3], fmt='theoretical log normal\n(mu=%#.3f, sig=%#.3f)'), y = 'snow depth distribution') +
  theme_minimal()

p_107 = ggplot(data_hs, aes(sample=hs)) +
  stat_qq(distribution=stats::qlnorm, na.rm=TRUE, dparams=c(mu[4], sig[4])) +
  geom_abline(intercept=0, slope=1) +
  xlim(0, 1) + 
  labs(title=datelist[4],  x=sprintf(mu[4], sig[4], fmt='theoretical log normal\n(mu=%#.3f, sig=%#.3f)'), y = 'snow depth distribution') +
  theme_minimal()

p_123 = ggplot(data_hs, aes(sample=hs)) +
  stat_qq(distribution=stats::qlnorm, na.rm=TRUE, dparams=c(mu[5], sig[5])) +
  geom_abline(intercept=0, slope=1) +
  xlim(0, 1) + 
  labs(title=datelist[5],  x=sprintf(mu[5], sig[5], fmt='theoretical log normal\n(mu=%#.3f, sig=%#.3f)'), y = 'snow depth distribution') +
  theme_minimal()

p_qq = grid.arrange(p_045, p_050, p_052, p_107, p_123, nrow=1, top = textGrob('Quantile-Quantile plots of snow depth distributions vs. parameter-fit log-normal'))

p_hist = ggplot(data_hs, aes(x=hs)) +
  facet_grid(. ~ date) +
  geom_histogram(aes(y=..density..), fill='blue', alpha=0.5, binwidth = .01) +
  geom_density(size=1) +
  xlim(-.2, .8) +
  labs(x='Snow depth (m)', y='frequency', title='Snow depth distributions for different survey dates') +
  theme_minimal()

p_theo = ggplot(data_hs) +
  facet_grid(. ~ date) +
  geom_histogram(aes(x=hs, y=..density..), fill='blue', alpha=0.5, binwidth = .01) +
  geom_density(aes(x=lnsamps), color='black', size=1) +
  xlim(-.2, .8) +
  labs(x='Snow depth (m)', y='frequency', title='Comparison of snow depth distribution with samples from theoretical log-normal') +
  theme_minimal()


p_sdd = grid.arrange(p_hist, p_theo, p_qq, nrow=3)
ggsave(paste0(gd, "snow_depth_log_normal_fitting.png"), p_sdd, width = plot_w, height = plot_h, units = "cm")


# ,gp=gpar(fontsize=20,font=3)