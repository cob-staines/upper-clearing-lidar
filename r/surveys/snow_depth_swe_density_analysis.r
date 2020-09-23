library('dplyr')
library('tidyr')
library('ggplot2')

# load survey samples
tasks <- c("19_045", "19_050", "19_052", "19_107", "19_123")

survey = data.frame()
for (ii in 1:5) {
  snow_file <- paste0("C:/Users/Cob/index/educational/usask/research/masters/data/surveys/depth_swe/marmot_snow_surveys_raw_", tasks[ii] , ".csv")
  temp = read.csv(snow_file, header=TRUE, na.strings = c("NA",""), skip=1, sep=",")
  temp$doy = tasks[ii]
  survey = rbind(survey, temp)
}

survey = survey[,c('snow_depth_cm', 'swe_raw_cm', 'swe_tare_cm', 'swe_quality_flag', 'standardized_survey_notes', 'doy')]

survey$swe_mm = 10 * (survey$swe_raw_cm - survey$swe_tare_cm)
survey$density = survey$swe_mm / (survey$snow_depth_cm * 0.01)

survey$swe_quality_flag[is.na(survey$swe_quality_flag)] = 0

survey = survey[!is.na(survey$swe_raw_cm),]
survey = survey[survey$swe_quality_flag == 0,]
# survey$day = paste0('19_', survey$doy)
# calculate density
# facet grid plot of density vs depth for all dates

ggplot(survey, aes(x=snow_depth_cm, y=swe_mm, color=standardized_survey_notes)) +
  facet_grid(. ~ doy) +
  geom_point()

ggplot(survey, aes(x=snow_depth_cm, y=density, color=standardized_survey_notes)) +
  facet_grid(. ~ doy) +
  geom_point()
  geom_smooth(method = "lm", se=FALSE, color="black", formula = y ~ x)

survey %>%
  filter(standardized_survey_notes == 'forest') %>%
  ggplot(., aes(x=snow_depth_cm, y=density)) +
    facet_grid(. ~ doy) +
    geom_point() +
    labs(title('Snow depth vs. density for forest points ove different days'))


survey %>%
  filter(doy %in% c('19_045', '19_050', '19_052')) %>%
  ggplot(., aes(x=snow_depth_cm, y=density, color=doy)) +
    facet_grid(. ~ standardized_survey_notes) +
    geom_point()
    theme_minimal()

# build forest linear models
f_045 = survey[(survey$standardized_survey_notes == 'forest') & (survey$doy == '19_045'),]
f_050 = survey[(survey$standardized_survey_notes == 'forest') & (survey$doy == '19_050'),]
f_052 = survey[(survey$standardized_survey_notes == 'forest') & (survey$doy == '19_052'),]
f_107 = survey[(survey$standardized_survey_notes == 'forest') & (survey$doy == '19_107'),]
f_123 = survey[(survey$standardized_survey_notes == 'forest') & (survey$doy == '19_123'),]

c_045 = survey[(survey$standardized_survey_notes == 'clearing') & (survey$doy == '19_045'),]
c_050 = survey[(survey$standardized_survey_notes == 'clearing') & (survey$doy == '19_050'),]
c_052 = survey[(survey$standardized_survey_notes == 'clearing') & (survey$doy == '19_052'),]
c_107 = survey[(survey$standardized_survey_notes == 'clearing') & (survey$doy == '19_107'),]
c_123 = survey[(survey$standardized_survey_notes == 'clearing') & (survey$doy == '19_123'),]

a_045 = survey[(survey$doy == '19_045'),]
a_050 = survey[(survey$doy == '19_050'),]
a_052 = survey[(survey$doy == '19_052'),]
a_107 = survey[(survey$doy == '19_107'),]
a_123 = survey[(survey$doy == '19_123'),]



lm_f_045 = lm(density ~ snow_depth_cm, data = f_045)
lm_f_050 = lm(density ~ snow_depth_cm, data = f_050)
lm_f_052 = lm(density ~ snow_depth_cm, data = f_052)
lm_f_107 = lm(density ~ snow_depth_cm, data = f_107)
lm_f_123 = lm(density ~ snow_depth_cm, data = f_123)

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

ks.test(f_045$density, f_050$density)
ks.test(f_050$density, f_052$density)
ks.test(f_045$density, f_052$density)

mean(f_045$density)
mean(f_050$density)
mean(f_052$density)
mean(f_107$density)
mean(f_123$density)
