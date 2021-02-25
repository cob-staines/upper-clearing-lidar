library('dplyr')
library('tidyr')
library('ggplot2')
library('grid')
library('gridExtra')

plot_out_dir = "C:/Users/Cob/index/educational/usask/research/masters/graphics/thesis_graphics/frequency distributions/swe_distribution_fitting/"
p_width = 8  # inches
p_height = 5.7  # inches
dpi = 100

d_045_in = 'C:/Users/Cob/index/educational/usask/research/masters/data/lidar/analysis/rejection_sampled/resampled_swe_19_045_ahpl_r.05m_interp2x_by_contact-number.csv'
d_045 = read.csv(d_045_in, header=TRUE, na.strings = c("NA",""), sep=",") %>%
  rename(swe = swe_19_045)

d_050_in = 'C:/Users/Cob/index/educational/usask/research/masters/data/lidar/analysis/rejection_sampled/resampled_swe_19_050_ahpl_r.05m_interp2x_by_contact-number.csv'
d_050 = read.csv(d_050_in, header=TRUE, na.strings = c("NA",""), sep=",") %>%
  rename(swe = swe_19_050)

d_052_in = 'C:/Users/Cob/index/educational/usask/research/masters/data/lidar/analysis/rejection_sampled/resampled_swe_19_052_ahpl_r.05m_interp2x_by_contact-number.csv'
d_052 = read.csv(d_052_in, header=TRUE, na.strings = c("NA",""), sep=",") %>%
  rename(swe = swe_19_052)


# coefficient of variation
cv_045 = sd(d_045$swe)/mean(d_045$swe)
cv_050 = sd(d_050$swe)/mean(d_050$swe)
cv_052 = sd(d_052$swe)/mean(d_052$swe)


# shook 1995
qq_func = function(data){
  swe = data$swe
  
  # calculate mean and standard deviation in transformed distribution
  s_y = sqrt(log(1 + (sd(swe)/mean(swe))^2))
  y_bar = log(mean(swe)) - s_y^2/2
  
  # rank by decreasing quantile
  p_ = rank(-1 * swe)/(nrow(data) + 1)
  f_ = 1 - p_
  
  # calculate k_value frequency factor
  k_y = qnorm(f_, mean=0, sd=1)
  k_ = (exp(s_y * k_y - (s_y^2)/2) - 1)/sqrt(exp(s_y^2) - 1)
  
  data$k_vals = k_
  
  data
}


k_045 = qq_func(d_045)
k_050 = qq_func(d_050)
k_052 = qq_func(d_052)



# 045
ggplot(k_045, aes(x=k_vals, y=swe)) +
  geom_point(size=.75) +
  labs(x='K', y='SWE (mm)', title='SWE - lognormal Q-Q plot for 14 Feb. 2019 (n=298831)\nUpper Forest, 5cm resolution, bias corrected with LAI') + 
  theme_minimal()
ggsave(paste0(plot_out_dir, "swe_19_045_ahpl_lognormal_qq.png"), width=p_width, height=p_height, dpi=dpi)

# 050
ggplot(k_050, aes(x=k_vals, y=swe)) +
  geom_point(size=.75) +
  labs(x='K', y='SWE (mm)', title='SWE - lognormal Q-Q plot for 19 Feb. 2019 (n=296589)\nUpper Forest, 5cm resolution, bias corrected with LAI') +
  theme_minimal()
ggsave(paste0(plot_out_dir, "swe_19_050_ahpl_lognormal_qq.png"), width=p_width, height=p_height, dpi=dpi)

# 052
ggplot(k_052, aes(x=k_vals, y=swe)) +
  geom_point(size=.75) +
  labs(x='K', y='SWE (mm)', title='SWE - lognormal Q-Q plot for 21 Feb. 2019 (n=174647)\nUpper Forest, 5cm resolution, bias corrected with LAI') +
  theme_minimal()
ggsave(paste0(plot_out_dir, "swe_19_052_ahpl_lognormal_qq.png"), width=p_width, height=p_height, dpi=dpi)


# linear fitting
lm_045 = lm(swe ~ k_vals, data=k_045)
lm_050 = lm(swe ~ k_vals, data=k_050)
lm_052 = lm(swe ~ k_vals, data=k_052)

summary(lm_045)
summary(lm_050)
summary(lm_052)


lognorm_model = function(k_data, lm){
  x_vals = seq(min(k_data$swe), max(k_data$swe), by = 0.1)

  mu = summary(lm)$coefficients[1]
  sig = summary(lm)$coefficients[2]
  
  lmu = log(mu^2/sqrt(mu^2 + sig^2))
  lsig = sqrt(log(1 + sig^2/mu^2))
  
  y_vals = dlnorm(x_vals, meanlog=lmu, sdlog=lsig)
  m_data = data.frame(x_vals, y_vals)
}

m_045 = lognorm_model(k_045, lm_045)
ggplot(k_045, aes(x=swe)) +
  geom_histogram(aes(y=..density..), fill='cornflowerblue', alpha=1, binwidth = 1) +
  geom_line(data=m_045, aes(x=x_vals, y=y_vals), size=1) +
  labs(x='SWE [mm]', y='Density [-]', title='SWE observed and lognormal modeled distributions for\n14 Feb. 2019 Upper Forest, 5cm resolution, bias corrected')
ggsave(paste0(plot_out_dir, "swe_19_045_ahpl_lognormal_model.png"), width=p_width, height=p_height, dpi=dpi)

m_050 = lognorm_model(k_050, lm_050)
ggplot(k_050, aes(x=swe)) +
  geom_histogram(aes(y=..density..), fill='cornflowerblue', alpha=1, binwidth = 1) +
  geom_line(data=m_050, aes(x=x_vals, y=y_vals), size=1) +
  labs(x='SWE [mm]', y='Density [-]', title='SWE observed and lognormal modeled distributions for\n19 Feb. 2019 Upper Forest, 5cm resolution, bias corrected')
ggsave(paste0(plot_out_dir, "swe_19_050_ahpl_lognormal_model.png"), width=p_width, height=p_height, dpi=dpi)

m_052 = lognorm_model(k_052, lm_052)
ggplot(k_052, aes(x=swe)) +
  geom_histogram(aes(y=..density..), fill='cornflowerblue', alpha=1, binwidth = 1) +
  geom_line(data=m_052, aes(x=x_vals, y=y_vals), size=1) +
  labs(x='SWE [mm]', y='Density [-]', title='SWE observed and lognormal modeled distributions for\n21 Feb. 2019 Upper Forest, 5cm resolution, bias corrected')
ggsave(paste0(plot_out_dir, "swe_19_052_ahpl_lognormal_model.png"), width=p_width, height=p_height, dpi=dpi)

# 
# ### old hat below
# 
# data_in = 'C:/Users/Cob/index/educational/usask/research/masters/data/lidar/analysis/swe_uf_.25m_canopy_19_149.csv'
# data = read.csv(data_in, header=TRUE, na.strings = c("NA",""), sep=",")
# 
# # filter to upper forest
# data = data[data$uf == 1, ]
# 
# 
# gd = 'C:/Users/Cob/index/educational/usask/research/masters/graphics/automated/'
# plot_h = 21  # cm
# plot_w = 29.7  # cm
# 
# # gather snow depths  
# data_swe = data %>%
#   gather("date", "swe", c(8, 10, 12, 14, 16))
# data_swe$date = as.factor(data_swe$date)
# levels(data_swe$date) = c("19_045", "19_050", "19_052", "19_107", "19_123")
# 
# 
# 
# # calculate K
# datelist = levels(data_swe$date)
# data_swe$k_vals = NA
# date = datelist[3]
# for (date in levels(data_swe$date)){
#   # pull out non-negative and non-NA swe values for the corresponding date
#   valid_samps = (data_swe$date == date) & (data_swe$swe > 0) & !is.na(data_swe$swe)
#   swe = data_swe$swe[valid_samps]
#   
#   # calculate mean and standard deviation in transformed distribution
#   s_y = sqrt(log(1 + (sd(swe)/mean(swe))^2))
#   y_bar = log(mean(swe)) - s_y^2/2
#   
#   # rank by decreasing quantile
#   p_ = rank(-1 * swe)/(sum(valid_samps) + 1)
#   f_ = 1 - p_
#   
#   # calculate k_value frequency factor
#   k_y = qnorm(f_, mean=0, sd=1)
#   k_ = (exp(s_y * k_y - (s_y^2)/2) - 1)/sqrt(exp(s_y^2) - 1)
#   
#   data_swe$k_vals[valid_samps] = k_
# }
# 
# # plots
# 
# p_hist = ggplot(data_swe, aes(x=swe)) +
#   facet_grid(. ~ date) +
#   geom_histogram(aes(y=..density..), fill='blue', alpha=0.5, binwidth = 1) +
#   labs(x='SWE (mm)', y='frequency', title='SWE distributions within the Upper Forest for different survey dates') +
#   theme_minimal()
# 
# p_swek = ggplot(data_swe, aes(x=k_vals, y=swe)) +
#   facet_grid(. ~ date) +
#   geom_line(size=.75) +
#   labs(x='K', y='SWE (mm)', title='SWE vs. lognormal frequency factor K over the Upper Forest\nslope = sd, intercept = mean of lognormal distribution (Shook 1995)') +
#   ylim(0, 255)
# 
# 
# 
# p_sk = grid.arrange(p_hist, p_swek, nrow=3)
# ggsave(paste0(gd, "swe_dist.pdf"), p_sk, width = plot_w, height = plot_h, units = "cm")
# 
# 
# 
# ##### Everything below is old hat #####
# 
# datelist = levels(data_hs$date)
# mu <- numeric(length = length(datelist))
# sig <- numeric(length = length(datelist))
# data_swe$lnsamps = 0
# for (ii in 1:length(datelist)){
#   date = datelist[ii]
#   samples = data_swe$swe[data_swe$date == date]
#   mu_x = mean(samples, na.rm=TRUE)
#   sig_x = sd(samples, na.rm=TRUE)
#   
#   mu[ii] = log(mu_x^2/sqrt(mu_x^2 + sig_x^2))
#   sig[ii] = sqrt(log(1 + sig_x^2/mu_x^2))
#   
#   n = length(samples)
#   data_swe$lnsamps[data_swe$date == date] = rlnorm(n, meanlog=mu[ii], sdlog=sig[ii])
# }
# 
# 
# p_045 = ggplot(data_hs, aes(sample=hs)) +
#   stat_qq(distribution=stats::qlnorm, na.rm=TRUE, dparams=c(mu[1], sig[1])) +
#   geom_abline(intercept=0, slope=1) +
#   xlim(0, 1) + 
#   labs(title=datelist[1],  x=sprintf(mu[1], sig[1], fmt='theoretical log normal\n(mu=%#.3f, sig=%#.3f)'), y = 'snow depth distribution') +
#   theme_minimal()
# 
# p_050 = ggplot(data_hs, aes(sample=hs)) +
#   stat_qq(distribution=stats::qlnorm, na.rm=TRUE, dparams=c(mu[2], sig[2])) +
#   geom_abline(intercept=0, slope=1) +
#   xlim(0, 1) + 
#   labs(title=datelist[2],  x=sprintf(mu[2], sig[2], fmt='theoretical log normal\n(mu=%#.3f, sig=%#.3f)'), y = 'snow depth distribution') +
#   theme_minimal()
# 
# p_052 = ggplot(data_hs, aes(sample=hs)) +
#   stat_qq(distribution=stats::qlnorm, na.rm=TRUE, dparams=c(mu[3], sig[3])) +
#   geom_abline(intercept=0, slope=1) +
#   xlim(0, 1) + 
#   labs(title=datelist[3],  x=sprintf(mu[3], sig[3], fmt='theoretical log normal\n(mu=%#.3f, sig=%#.3f)'), y = 'snow depth distribution') +
#   theme_minimal()
# 
# p_107 = ggplot(data_hs, aes(sample=hs)) +
#   stat_qq(distribution=stats::qlnorm, na.rm=TRUE, dparams=c(mu[4], sig[4])) +
#   geom_abline(intercept=0, slope=1) +
#   xlim(0, 1) + 
#   labs(title=datelist[4],  x=sprintf(mu[4], sig[4], fmt='theoretical log normal\n(mu=%#.3f, sig=%#.3f)'), y = 'snow depth distribution') +
#   theme_minimal()
# 
# p_123 = ggplot(data_hs, aes(sample=hs)) +
#   stat_qq(distribution=stats::qlnorm, na.rm=TRUE, dparams=c(mu[5], sig[5])) +
#   geom_abline(intercept=0, slope=1) +
#   xlim(0, 1) + 
#   labs(title=datelist[5],  x=sprintf(mu[5], sig[5], fmt='theoretical log normal\n(mu=%#.3f, sig=%#.3f)'), y = 'snow depth distribution') +
#   theme_minimal()
# 
# p_qq = grid.arrange(p_045, p_050, p_052, p_107, p_123, nrow=1, top = textGrob('Quantile-Quantile plots of snow depth distributions vs. parameter-fit log-normal'))
# 
# p_hist = ggplot(data_hs, aes(x=hs)) +
#   facet_grid(. ~ date) +
#   geom_histogram(aes(y=..density..), fill='blue', alpha=0.5, binwidth = .01) +
#   geom_density(size=1) +
#   xlim(-.2, .8) +
#   labs(x='Snow depth (m)', y='frequency', title='Snow depth distributions for different survey dates') +
#   theme_minimal()
# 
# p_theo = ggplot(data_hs) +
#   facet_grid(. ~ date) +
#   geom_histogram(aes(x=hs, y=..density..), fill='blue', alpha=0.5, binwidth = .01) +
#   geom_density(aes(x=lnsamps), color='black', size=1) +
#   xlim(-.2, .8) +
#   labs(x='Snow depth (m)', y='frequency', title='Comparison of snow depth distribution with samples from theoretical log-normal') +
#   theme_minimal()
# 
# 
# p_sdd = grid.arrange(p_hist, p_theo, p_qq, nrow=3)
# ggsave(paste0(gd, "snow_depth_log_normal_fitting.png"), p_sdd, width = plot_w, height = plot_h, units = "cm")
# 
# 
# # ,gp=gpar(fontsize=20,font=3)