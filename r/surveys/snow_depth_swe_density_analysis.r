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

# drop unnecesary columns
survey = survey[,c('snow_depth_cm', 'swe_raw_cm', 'swe_tare_cm', 'swe_quality_flag', 'doy', 'swe_mm', 'density', 'cover', 'swe_quality_flag')]
# filter to qc'ed swe entries
survey = survey[!is.na(survey$swe_raw_cm),]
survey = survey[survey$swe_quality_flag == 0,]
# filter to depths greater than 20cm
survey = survey[survey$snow_depth_cm >= 20,]


p_swe = ggplot(survey, aes(x=snow_depth_cm, y=swe_mm, color=cover)) +
  facet_grid(. ~ doy) +
  geom_point() +
  labs(title='Snow depth vs. SWE across survey days', x='Snow depth (cm)', y='SWE (mm)') +
  xlim(0, NA) +
  ylim(0, NA)

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
# ggsave(paste0(gd, "snow_depth_v_density.pdf"), p_sden, width = 29.7, height = 21, units = "cm")

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

for_cle_all %>%
  filter(doy %in% c("19_045", "19_050", "19_052")) %>%
  ggplot(., aes(x=snow_depth_cm, y=density)) +
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

a_4550 = survey[(survey$doy %in% c('19_045', '19_050')),]
a_5052 = survey[(survey$doy %in% c('19_050', '19_052')),]
a_5207 = survey[(survey$doy %in% c('19_052', '19_107')),]
a_0723 = survey[(survey$doy %in% c('19_107', '19_123')),]

a_455052 = survey[(survey$doy %in% c('19_045', '19_050', '19_052')),]





# forest linear density
lm_flin_045 = lm(density ~ snow_depth_cm, data = f_045)
lm_flin_050 = lm(density ~ snow_depth_cm, data = f_050)
lm_flin_052 = lm(density ~ snow_depth_cm, data = f_052)
lm_flin_107 = lm(density ~ snow_depth_cm, data = f_107)
lm_flin_123 = lm(density ~ snow_depth_cm, data = f_123)

# forest constant density
lm_fc_045 = lm(swe_mm ~ 0 + snow_depth_cm, data = f_045)
lm_fc_050 = lm(swe_mm ~ 0 + snow_depth_cm, data = f_050)
lm_fc_052 = lm(swe_mm ~ 0 + snow_depth_cm, data = f_052)
lm_fc_107 = lm(swe_mm ~ 0 + snow_depth_cm, data = f_107)
lm_fc_123 = lm(swe_mm ~ 0 + snow_depth_cm, data = f_123)

# all linear density
lm_alin_045 = lm(density ~ snow_depth_cm, data = a_045)
lm_alin_050 = lm(density ~ snow_depth_cm, data = a_050)
lm_alin_052 = lm(density ~ snow_depth_cm, data = a_052)
lm_alin_107 = lm(density ~ snow_depth_cm, data = a_107)
lm_alin_123 = lm(density ~ snow_depth_cm, data = a_123)

# all constant density
lm_ac_045 = lm(swe_mm ~ 0 + snow_depth_cm, data = a_045)
lm_ac_050 = lm(swe_mm ~ 0 + snow_depth_cm, data = a_050)
lm_ac_052 = lm(swe_mm ~ 0 + snow_depth_cm, data = a_052)
lm_ac_107 = lm(swe_mm ~ 0 + snow_depth_cm, data = a_107)
lm_ac_123 = lm(swe_mm ~ 0 + snow_depth_cm, data = a_123)

# adjacent combined
lm_acli_4550 = lm(density ~ snow_depth_cm, data = a_4550)
lm_acli_5052 = lm(density ~ snow_depth_cm, data = a_5052)
lm_acli_5207 = lm(density ~ snow_depth_cm, data = a_5207)
lm_acli_0723 = lm(density ~ snow_depth_cm, data = a_0723)

summary(lm_acli_5052)

# all combined linear density
lm_alin_455052 = lm(density ~ snow_depth_cm, data = a_455052)

# define model funtion
# linear density
swelindensfunc <- function(lmobj, hs) {
  swe = hs * (summary(lmobj)$coefficients[2] * hs + summary(lmobj)$coefficients[1]) * 0.01
  swe
}
# constant density
swecdensfunc <- function(lmobj, hs) {
  swe = summary(lmobj)$coefficients[1] * hs
  swe
}
# exponential density
sweexpdensfunc <- function(nls_obj, hs) {
  swe = hs * predict(nls_obj, newdata=hs) * 0.01
  swe
}

# now plot it out
# forest linear
survey_c = survey
survey_c$swe_flin = NA
survey_c$swe_flin[survey_c$doy == "19_045"] = swelindensfunc(lm_flin_045, survey_c$snow_depth_cm[survey_c$doy == "19_045"])
survey_c$swe_flin[survey_c$doy == "19_050"] = swelindensfunc(lm_flin_050, survey_c$snow_depth_cm[survey_c$doy == "19_050"])
survey_c$swe_flin[survey_c$doy == "19_052"] = swelindensfunc(lm_flin_052, survey_c$snow_depth_cm[survey_c$doy == "19_052"])
survey_c$swe_flin[survey_c$doy == "19_107"] = swelindensfunc(lm_flin_107, survey_c$snow_depth_cm[survey_c$doy == "19_107"])
survey_c$swe_flin[survey_c$doy == "19_123"] = swelindensfunc(lm_flin_123, survey_c$snow_depth_cm[survey_c$doy == "19_123"])

survey_c$swe_fc = NA
survey_c$swe_fc[survey_c$doy == "19_045"] = swecdensfunc(lm_fc_045, survey_c$snow_depth_cm[survey_c$doy == "19_045"])
survey_c$swe_fc[survey_c$doy == "19_050"] = swecdensfunc(lm_fc_050, survey_c$snow_depth_cm[survey_c$doy == "19_050"])
survey_c$swe_fc[survey_c$doy == "19_052"] = swecdensfunc(lm_fc_052, survey_c$snow_depth_cm[survey_c$doy == "19_052"])
survey_c$swe_fc[survey_c$doy == "19_107"] = swecdensfunc(lm_fc_107, survey_c$snow_depth_cm[survey_c$doy == "19_107"])
survey_c$swe_fc[survey_c$doy == "19_123"] = swecdensfunc(lm_fc_123, survey_c$snow_depth_cm[survey_c$doy == "19_123"])

survey_c$swe_alin = NA
survey_c$swe_alin[survey_c$doy == "19_045"] = swelindensfunc(lm_alin_045, survey_c$snow_depth_cm[survey_c$doy == "19_045"])
survey_c$swe_alin[survey_c$doy == "19_050"] = swelindensfunc(lm_alin_050, survey_c$snow_depth_cm[survey_c$doy == "19_050"])
survey_c$swe_alin[survey_c$doy == "19_052"] = swelindensfunc(lm_alin_052, survey_c$snow_depth_cm[survey_c$doy == "19_052"])
survey_c$swe_alin[survey_c$doy == "19_107"] = swelindensfunc(lm_alin_107, survey_c$snow_depth_cm[survey_c$doy == "19_107"])
survey_c$swe_alin[survey_c$doy == "19_123"] = swelindensfunc(lm_alin_123, survey_c$snow_depth_cm[survey_c$doy == "19_123"])

survey_c$swe_ac = NA
survey_c$swe_ac[survey_c$doy == "19_045"] = swecdensfunc(lm_ac_045, survey_c$snow_depth_cm[survey_c$doy == "19_045"])
survey_c$swe_ac[survey_c$doy == "19_050"] = swecdensfunc(lm_ac_050, survey_c$snow_depth_cm[survey_c$doy == "19_050"])
survey_c$swe_ac[survey_c$doy == "19_052"] = swecdensfunc(lm_ac_052, survey_c$snow_depth_cm[survey_c$doy == "19_052"])
survey_c$swe_ac[survey_c$doy == "19_107"] = swecdensfunc(lm_ac_107, survey_c$snow_depth_cm[survey_c$doy == "19_107"])
survey_c$swe_ac[survey_c$doy == "19_123"] = swecdensfunc(lm_ac_123, survey_c$snow_depth_cm[survey_c$doy == "19_123"])

survey_c$swe_ax = NA
survey_c$swe_ax[survey_c$doy == "19_045"] = sweexpdensfunc(nls_a_045, survey_c$snow_depth_cm[survey_c$doy == "19_045"])
survey_c$swe_ax[survey_c$doy == "19_050"] = sweexpdensfunc(nls_a_050, survey_c$snow_depth_cm[survey_c$doy == "19_050"])
survey_c$swe_ax[survey_c$doy == "19_052"] = sweexpdensfunc(nls_a_052, survey_c$snow_depth_cm[survey_c$doy == "19_052"])
survey_c$swe_ax[survey_c$doy == "19_107"] = sweexpdensfunc(nls_a_107, survey_c$snow_depth_cm[survey_c$doy == "19_107"])
survey_c$swe_ax[survey_c$doy == "19_123"] = sweexpdensfunc(nls_a_123, survey_c$snow_depth_cm[survey_c$doy == "19_123"])

ggplot(survey_c, aes(x=snow_depth_cm, y=swe_mm)) +
  facet_grid(. ~ doy) +
  geom_point() +
  geom_line(aes(x = snow_depth_cm, y=swe_flin, color="f_linear")) +
  geom_line(aes(x = snow_depth_cm, y=swe_fc, color="f_const")) +
  geom_line(aes(x = snow_depth_cm, y=swe_alin, color="a_linear")) +
  geom_line(aes(x = snow_depth_cm, y=swe_ac, color="a_const")) +
  # geom_line(aes(x = snow_depth_cm, y=swe_ax, color="a_exp")) +
  labs(title='Snow depth vs. SWE across survey days', x='Snow depth (cm)', y='SWE (mm)') +
  xlim(0, NA) +
  ylim(0, NA)



# non linear SWE models
library('nls.multstart')

# fixed intercept PomGray
ii = 67.92

nls_a_045 <- nls_multstart(density ~ 67.92 + c/b  - (1 - exp(-snow_depth_cm * c))/(b * snow_depth_cm),
                           data = a_045,
                           lower=c(b=0, c=-10),
                           upper=c(b=1, c=10),
                           start_lower = c(b=0, c=-10),
                           start_upper = c(b=1, c=10),
                           iter = 500,
                           supp_errors = "Y")

nls_a_050 <- nls_multstart(density ~ 67.92 + c/b  - (1 - exp(-snow_depth_cm * c))/(b * snow_depth_cm),
                           data = a_050,
                           lower=c(b=0, c=-10),
                           upper=c(b=1, c=10),
                           start_lower = c(b=0, c=-10),
                           start_upper = c(b=1, c=10),
                           iter = 500,
                           supp_errors = "Y")

nls_a_052 <- nls_multstart(density ~ 67.92 + c/b  - (1 - exp(-snow_depth_cm * c))/(b * snow_depth_cm),
                           data = a_052,
                           lower=c(b=0, c=-10),
                           upper=c(b=1, c=10),
                           start_lower = c(b=0, c=-10),
                           start_upper = c(b=1, c=10),
                           iter = 500,
                           supp_errors = "Y")

nls_a_107 <- nls_multstart(density ~ 67.92 + c/b  - (1 - exp(-snow_depth_cm * c))/(b * snow_depth_cm),
                           data = a_107,
                           lower=c(b=0, c=-10),
                           upper=c(b=1, c=10),
                           start_lower = c(b=0, c=-10),
                           start_upper = c(b=1, c=10),
                           iter = 500,
                           supp_errors = "Y")

nls_a_123 <- nls_multstart(density ~ 67.92 + c/b  - (1 - exp(-snow_depth_cm * c))/(b * snow_depth_cm),
                           data = a_123,
                           lower=c(b=0, c=-10),
                           upper=c(b=1, c=10),
                           start_lower = c(b=0, c=-10),
                           start_upper = c(b=1, c=10),
                           iter = 500,
                           supp_errors = "Y")


nls_a_5052 <- nls_multstart(density ~ 67.92 + c/b  - (1 - exp(-snow_depth_cm * c))/(b * snow_depth_cm),
                              data = a_5052,
                              lower=c(b=0, c=-10),
                              upper=c(b=1, c=10),
                              start_lower = c(b=0, c=-10),
                              start_upper = c(b=1, c=10),
                              iter = 500,
                              supp_errors = "Y")


nls_a_455052 <- nls_multstart(density ~ 67.92 + c/b  - (1 - exp(-snow_depth_cm * c))/(b * snow_depth_cm),
                           data = a_455052,
                           lower=c(b=0, c=-10),
                           upper=c(b=1, c=10),
                           start_lower = c(b=0, c=-10),
                           start_upper = c(b=1, c=10),
                           iter = 500,
                           supp_errors = "Y")


summary(nls_a_045)
summary(nls_a_050)
summary(nls_a_052)
summary(nls_a_107)
summary(nls_a_123)
summary(nls_a_5052)
summary(nls_a_455052)

plot_nls <- function(nls_object, data) {
  predframe <- tibble(snow_depth_cm=seq(from=0, to=max(data$snow_depth_cm), length.out = 1024)) %>%
    mutate(density = predict(nls_object, newdata = list(snow_depth_cm=.$snow_depth_cm)))
  ggplot(data, aes(x=snow_depth_cm, y=density)) +
    geom_point(size=3) +
    geom_line(data = predframe, aes(x=snow_depth_cm, y=density)) +
    xlim(0, max(data$snow_depth_cm * 1.05)) +
    ylim(0, max(data$density * 1.05))
}


ggplot(a_455052, aes(x=snow_depth_cm, y=density, color=doy)) +
  geom_point(size=3)
  geom_line(data = predframe, aes(x=snow_depth_cm, y=density)) +
  xlim(0, max(data$snow_depth_cm * 1.05)) +
  ylim(0, max(data$density * 1.05))

plot_nls(nls_a_045, a_045)
plot_nls(nls_a_050, a_050)
plot_nls(nls_a_052, a_052)
plot_nls(nls_a_107, a_107)
plot_nls(nls_a_123, a_123)
plot_nls(nls_a_5052, a_5052)
plot_nls(nls_a_455052, a_455052)

plot_nls_swe <- function(nls_object, data) {
  predframe <- tibble(snow_depth_cm=seq(from=0, to=max(data$snow_depth_cm), length.out = 1024)) %>%
    mutate(swe_mmm = snow_depth_cm * predict(nls_object, newdata = list(snow_depth_cm=.$snow_depth_cm)) / 100)
  ggplot(data, aes(x=snow_depth_cm, y=swe_mm)) +
    geom_point(size=3) +
    geom_line(data = predframe, aes(x=snow_depth_cm, y=swe_mmm)) +
    xlim(0, max(data$snow_depth_cm * 1.05)) +
    ylim(0, max(data$density * 1.05))
}

plot_nls_swe(nls_a_455052, a_455052)

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
