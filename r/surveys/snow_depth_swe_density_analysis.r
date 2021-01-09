library('dplyr')
library('tidyr')
library('ggplot2')
library('grid')
library('gridExtra')

# load survey samples
tasks <- c("19_045", "19_050", "19_052", "19_107", "19_123")

survey = data.frame()
for (ii in 1:5) {
  snow_file <- paste0("C:/Users/Cob/index/educational/usask/research/masters/data/surveys/depth_swe/marmot_snow_surveys_raw_", tasks[ii] , ".csv")
  temp = read.csv(snow_file, header=TRUE, na.strings = c("NA",""), skip=1, sep=",")
  temp$doy = tasks[ii]
  survey = rbind(survey, temp)
}

survey$swe_mm = 10 * (survey$swe_raw_cm - survey$swe_tare_cm)
survey$density = survey$swe_mm / (survey$snow_depth_cm * 0.01)
survey$cover = survey$standardized_survey_notes

survey$swe_quality_flag[is.na(survey$swe_quality_flag)] = 0

ggplot(survey, aes(x=snow_depth_cm, y=density, color=as.factor(swe_quality_flag))) +
  facet_grid(doy ~ .) +
  geom_point()

survey = survey[,c('snow_depth_cm', 'swe_raw_cm', 'swe_tare_cm', 'swe_quality_flag', 'doy', 'swe_mm', 'density', 'cover', 'swe_quality_flag')]

survey = survey[!is.na(survey$swe_raw_cm),]
survey = survey[survey$swe_quality_flag == 0,]
# survey$day = paste0('19_', survey$doy)
# calculate density
# facet grid plot of density vs depth for all dates


p_swe = ggplot(survey, aes(x=snow_depth_cm, y=swe_mm, color=cover)) +
  facet_grid(. ~ doy) +
  geom_point() +
  labs(title='Snow depth vs. SWE across survey days', x='Snow depth (cm)', y='SWE (mm)')

p_den = ggplot(survey, aes(x=snow_depth_cm, y=density, color=cover)) +
  facet_grid(. ~ doy) +
  geom_point() +
  labs(title='Snow depth vs. density across survey days', x='Snow depth (cm)', y='Density (kg/m^3)')

foreststuff =  survey %>%
  filter(cover == 'forest')

p_for = ggplot(foreststuff, aes(x=snow_depth_cm, y=density, color=cover)) +
  facet_grid(. ~ doy) +
  geom_point() +
  labs(title='Snow depth vs. density for forest points across survey days', x='Snow depth (cm)', y='Density (kg/m^3)') +
  geom_smooth(method = "lm", se=FALSE, color="black", formula = y ~ x)

p_sden = grid.arrange(p_swe, p_den, p_for, nrow=3)
gd = 'C:/Users/Cob/index/educational/usask/research/masters/graphics/automated/'
ggsave(paste0(gd, "snow_depth_v_density.pdf"), p_sden, width = 29.7, height = 21, units = "cm")

survey %>%
  filter(doy %in% c("19_045", "19_050", "19_052")) %>%
  ggplot(., aes(x=snow_depth_cm, y=density, color=as.factor(doy))) +
    facet_grid(. ~ cover) +
    geom_point() +
    geom_smooth(method='lm', formula=y~x)

all_vals = survey
all_vals$cover = 'all'

for_all = rbind(survey[survey$cover == 'forest',], all_vals)

for_cle_all = rbind(survey[survey$cover == 'clearing',], for_all)

ggplot(for_cle_all, aes(x=snow_depth_cm, y=density)) +
  facet_grid(cover ~ doy) +
  geom_point() +
  ylim(0, 350) +
  xlim(0, 85) +
  geom_smooth(method='lm', formula= y~x)

for_cle_all %>%
  filter(doy %in% c("19_045", "19_050", "19_052")) %>%
  ggplot(., aes(x=snow_depth_cm, y=density, color=doy)) +
    facet_grid(cover ~ .) +
    geom_point() +
    ylim(0, 350) +
    xlim(0, 85) +
    geom_smooth(method='lm', formula= y~x)



ggplot(for_all, aes(x=snow_depth_cm, y=swe_mm)) +
  facet_grid(cover ~ doy) +
  geom_point() +
  geom_smooth(method='lm', formula= y~ 0 + x)
    
## Linear models
# build forest linear models
f_045 = survey[(survey$cover == 'forest') & (survey$doy == '19_045'),]
f_050 = survey[(survey$cover == 'forest') & (survey$doy == '19_050'),]
f_052 = survey[(survey$cover == 'forest') & (survey$doy == '19_052'),]
f_107 = survey[(survey$cover == 'forest') & (survey$doy == '19_107'),]
f_123 = survey[(survey$cover == 'forest') & (survey$doy == '19_123'),]

c_045 = survey[(survey$cover == 'clearing') & (survey$doy == '19_045'),]
c_050 = survey[(survey$cover == 'clearing') & (survey$doy == '19_050'),]
c_052 = survey[(survey$cover == 'clearing') & (survey$doy == '19_052'),]
c_107 = survey[(survey$cover == 'clearing') & (survey$doy == '19_107'),]
c_123 = survey[(survey$cover == 'clearing') & (survey$doy == '19_123'),]

a_045 = survey[(survey$doy == '19_045'),]
a_050 = survey[(survey$doy == '19_050'),]
a_052 = survey[(survey$doy == '19_052'),]
a_107 = survey[(survey$doy == '19_107'),]
a_123 = survey[(survey$doy == '19_123'),]

a_455052 = survey[(survey$doy %in% c('19_045', '19_050', '19_052')),]
lm_a_455052 = lm(density ~ snow_depth_cm, data = a_455052)
summary(lm_a_455052)



lm_f_045 = lm(density ~ snow_depth_cm, data = f_045)
lm_f_050 = lm(density ~ snow_depth_cm, data = f_050)
lm_f_052 = lm(density ~ snow_depth_cm, data = f_052)
lm_f_107 = lm(density ~ snow_depth_cm, data = f_107)
lm_f_123 = lm(density ~ snow_depth_cm, data = f_123)

lm_f_045 = lm(density ~ 0 + swe_mm, data = f_045)
lm_f_050 = lm(density ~ 0 + swe_mm, data = f_050)
lm_f_052 = lm(density ~ 0 + swe_mm, data = f_052)
lm_f_107 = lm(density ~ 0 + swe_mm, data = f_107)
lm_f_123 = lm(density ~ 0 + swe_mm, data = f_123)

lm_c_045 = lm(density ~ snow_depth_cm, data = c_045)
lm_c_050 = lm(density ~ snow_depth_cm, data = c_050)
lm_c_052 = lm(density ~ snow_depth_cm, data = c_052)
lm_c_107 = lm(density ~ snow_depth_cm, data = c_107)
lm_c_123 = lm(density ~ snow_depth_cm, data = c_123)

lm_a_045 = lm(density ~ snow_depth_cm, data = a_045)
lm_a_050 = lm(density ~ snow_depth_cm, data = a_050)
lm_a_052 = lm(density ~ snow_depth_cm, data = a_052)
lm_a_107 = lm(density ~ snow_depth_cm, data = a_107)
lm_a_123 = lm(density ~ snow_depth_cm, data = a_123)

lm_a_045 = lm(density ~ 0 + swe_mm, data = a_045)
lm_a_050 = lm(density ~ 0 + swe_mm, data = a_050)
lm_a_052 = lm(density ~ 0 + swe_mm, data = a_052)
lm_a_107 = lm(density ~ 0 + swe_mm, data = a_107)
lm_a_123 = lm(density ~ 0 + swe_mm, data = a_123)

summary(lm_f_045)
summary(lm_f_050)
summary(lm_f_052)
summary(lm_f_107)
summary(lm_f_123)

summary(lm_c_045)
summary(lm_c_050)
summary(lm_c_052)
summary(lm_c_107)
summary(lm_c_123)

summary(lm_a_045)
summary(lm_a_050)
summary(lm_a_052)
summary(lm_a_107)
summary(lm_a_123)

# non linear SWE models
library('nls.multstart')

powfunc <- function(hs, a, b, c){
  a * hs ^ b + c
}

p_045 = a_045[a_045$snow_depth_cm > 20, ]

nls_a_045 <- nls_multstart(density ~ a  - (b / snow_depth_cm) * (1 - exp(snow_depth_cm * c)),
                           data = a_045,
                           lower=c(a=-1000, b=0, c=-10),
                           upper=c(a=1000, b=1000000, c=10),
                           start_lower = c(a=-1000, b=10000, c=-10),
                           start_upper = c(a=1000, b=30000, c=10),
                           iter = 500,
                           supp_errors = "Y")

nls_a_050 <- nls_multstart(density ~ a  - (b / snow_depth_cm) * (1 - exp(snow_depth_cm * c)),
                           data = a_050,
                           lower=c(a=-1000, b=0, c=-10),
                           upper=c(a=1000, b=1000000, c=10),
                           start_lower = c(a=-1000, b=10000, c=-10),
                           start_upper = c(a=1000, b=30000, c=10),
                           iter = 500,
                           supp_errors = "Y")

nls_a_052 <- nls_multstart(density ~ a  - (b / snow_depth_cm) * (1 - exp(snow_depth_cm * c)),
                           data = a_052,
                           lower=c(a=-1000, b=0, c=-10),
                           upper=c(a=1000, b=1000000, c=10),
                           start_lower = c(a=-1000, b=10000, c=-10),
                           start_upper = c(a=1000, b=30000, c=10),
                           iter = 500,
                           supp_errors = "Y")



nls_a_045 <- nls_multstart(density ~ a * (snow_depth_cm / 100) ^ b + c,
                           data = a_045,
                           lower=c(a=-100, b=-2, c=0),
                           upper=c(a=100, b=2, c=1000),
                           start_lower = c(a=-100, b=-2, c=0),
                           start_upper = c(a=100, b=2, c=1000),
                           iter = 500,
                           supp_errors = "Y")

nls_a_045 <- nls_multstart(density ~ a * (snow_depth_cm / 100) ^ b + c,
                           data = p_045,
                           lower=c(a=-100, b=-2, c=0),
                           upper=c(a=100, b=2, c=1000),
                           start_lower = c(a=-100, b=-3, c=0),
                           start_upper = c(a=100, b=2, c=1000),
                           iter = 500,
                           supp_errors = "Y")

nls_a_045 <- nls_multstart(density ~ b * snow_depth_cm  + c,
                           data = a_045,
                           lower=c(b=-3, c=0),
                           upper=c(b=3, c=1000),
                           start_lower = c(b=-3, c=0),
                           start_upper = c(b=2, c=1000),
                           iter = 500,
                           supp_errors = "Y")



summary(nls_a_045)
summary(nls_a_050)
summary(nls_a_052)

plot_nls <- function(nls_object, data) {
  predframe <- tibble(snow_depth_cm=seq(from=min(0), to=max(data$snow_depth_cm), length.out = 1024)) %>%
    mutate(density = predict(nls_object, newdata = list(snow_depth_cm=.$snow_depth_cm)))
  ggplot(data, aes(x=snow_depth_cm, y=density)) +
    geom_point(size=3) +
    geom_line(data = predframe, aes(x=snow_depth_cm, y=density)) +
    xlim(0, max(data$snow_depth_cm * 1.05)) +
    ylim(0, max(data$density * 1.05))
}

plot_nls(nls_a_045, a_045)
plot_nls(nls_a_050, a_050)
plot_nls(nls_a_052, a_052)

nlm_a_045 <- nls(density ~ a * snow_depth_cm ^ b + c, start=list(a=10, b=1, c=150), data = a_045, trace = TRUE)

nlm_a_045 <- nls(swe_mm ~ a * snow_depth_cm + b * (1 - exp(-snow_depth_cm/c)), start=list(a=7, b=-1177, c=200), data = a_045, trace = TRUE)
nlm_a_045 <- nls(swe_mm ~ a * snow_depth_cm + b * (1 - exp(-snow_depth_cm/c)), start=list(a=4.88, b=-204.7, c=67.3), data = a_045, trace = TRUE, control=nls.control(maxiter=1000, minFactor = 1/2048))
nlm_a_050 <- nls(swe_mm ~ a * snow_depth_cm + b * (1 - exp(-snow_depth_cm/c)), start=list(a=4.88, b=-204.7, c=67.3), data = a_050, trace=TRUE)
nlm_a_052 <- nls(swe_mm ~ a * snow_depth_cm + b * (1 - exp(-snow_depth_cm/c)), start=list(a=4.88, b=-204.7, c=67.3), data = a_052, trace=TRUE, control=nls.control(maxiter=1000))
nlm_a_107 <- nls(swe_mm ~ a * snow_depth_cm + b * (1 - exp(-snow_depth_cm/c)), start=list(a=4.88, b=-204.7, c=67.3), data = a_107)
nlm_a_123 <- nls(swe_mm ~ a * snow_depth_cm + b * (1 - exp(-snow_depth_cm/c)), start=list(a=4.88, b=-204.7, c=67.3), data = a_123)

print(1 - sum(resid(nlm_a_050)^2)/sum((a_050$swe_mm - mean(a_050$swe_mm))^2))

library(lme4)
mixed_lm_a_045 = lmer(density ~ snow_depth_cm + (1|standardized_survey_notes),data = a_045)
summary(mixed_lm_a_045)

ks.test(f_045$density, f_050$density)
ks.test(f_045$density, f_052$density)
ks.test(f_050$density, f_052$density)
ks.test(f_045$density, f_107$density)
ks.test(f_050$density, f_107$density)
ks.test(f_052$density, f_107$density)
ks.test(f_045$density, f_123$density)
ks.test(f_050$density, f_123$density)
ks.test(f_052$density, f_123$density)
ks.test(f_107$density, f_123$density)


mean(f_045$density)
mean(f_050$density)
mean(f_052$density)
mean(f_107$density)
mean(f_123$density)
