library('dplyr')
library('tidyr')
library('ggplot2')
library('stringr')

plot_out_dir = "C:/Users/Cob/index/educational/usask/research/masters/graphics/thesis_graphics/validation/lidar_swe_hs_validation/"
p_width = 8  # inches
p_height = 5.7  # inches
dpi = 100


hs_in = "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/analysis/validation/lidar_hs_point_samples_clean.csv"
hs = read.csv(hs_in, header=TRUE, na.strings = c("NA",""), sep=",")

swe_in = "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/analysis/validation/lidar_swe_point_samples.csv"
swe = read.csv(swe_in, header=TRUE, na.strings = c("NA",""), sep=",")

pd_in = "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/analysis/validation/lidar_point_density_point_samples.csv"
pd = read.csv(pd_in, header=TRUE, na.strings = c("NA",""), sep=",")

hs_parsed <- hs %>%
  gather('key', 'lidar_hs', 13:92) %>%
  mutate(doy=substr(key, 5, 7), lidar_res=as.numeric(substr(key, 9,11)), interp_len=str_sub(key,-1,-1)) %>%
  select(uid, doy, lidar_res, lidar_hs, interp_len)
hs_parsed$doy = as.numeric(hs_parsed$doy)

swe_parsed <- swe %>%
  gather('key', 'lidar_swe', 13:172) %>%
  mutate(doy=substr(key, 5, 7), lidar_res=as.numeric(substr(key, 9, 11)), density_assumption=str_sub(key,-6,-3), interp_len=str_sub(key,-1,-1)) %>%
  select(uid, doy, lidar_res, lidar_swe, density_assumption, interp_len)
swe_parsed$doy = as.numeric(swe_parsed$doy)
swe_parsed$density_assumption = as.factor(swe_parsed$density_assumption)

pd_parsed <- pd %>%
  gather('key', 'lidar_pd', 13:24) %>%
  mutate(doy=substr(key, 5, 7), lidar_res=as.numeric(substr(key, 9,11))) %>%
  select(uid, doy, lidar_res, lidar_pd)
pd_parsed$doy = as.numeric(pd_parsed$doy)



# load survey
survey_in = "C:/Users/Cob/index/educational/usask/research/masters/data/surveys/depth_swe/snow_survey_gnss_merged.csv"
survey = read.csv(survey_in, header=TRUE, na.strings = c("NA",""), sep=",")

survey$swe_mm = 10 * (survey$swe_raw_cm - survey$swe_tare_cm)
survey$density = survey$swe_mm / (survey$snow_depth_cm * 0.01)
survey$swe_quality_flag[is.na(survey$swe_quality_flag)] = 0
survey$vegetation = survey$cover
survey$vegetation[survey$vegetation == "edge"] = "forest"
survey$vegetation[survey$vegetation == "newtrees"] = "clearing"

# filter to first 3 days only
survey = filter(survey, doy %in% c(45, 50, 52))


# merge along uid
hs_merge <- merge(survey, hs_parsed, all.y = TRUE, by=c('uid', 'doy')) %>%
  filter(!is.na(Point.Id), !is.na(snow_depth_cm)) %>%
  mutate(hs_dif = lidar_hs - snow_depth_cm/100)

swe_merge <- merge(survey, swe_parsed, all.y = TRUE, by=c('uid', 'doy')) %>%
  filter(!is.na(Point.Id), !is.na(swe_mm)) %>%
  mutate(swe_dif = lidar_swe - swe_mm)
swe_merge = swe_merge[(swe_merge$cover == "clearing" & swe_merge$density_assumption == "clin") | (swe_merge$cover == "forest" & swe_merge$density_assumption == "fcon"), ]
swe_merge$swe_quality_flag[(swe_merge$doy == 45) & (swe_merge$cover == "forest")] = 1
swe_merge$swe_quality_flag[swe_merge$snow_depth_cm < 20] = 1
swe_merge = swe_merge[swe_merge$swe_quality_flag == 0,]
swe_merge = swe_merge[swe_merge$interp_len == 2,]

hs_swe = merge(hs_parsed, swe_parsed, all = TRUE, by=c('uid', 'doy', 'lidar_res', 'interp_len'))
hs_swe_pd = merge(pd_parsed, hs_swe, all.x = TRUE, by=c('uid', 'doy', 'lidar_res'))
all_merge = merge(survey, hs_swe_pd, all.y = TRUE, by=c('uid', 'doy')) %>%
  filter(!is.na(Point.Id), !is.na(swe_mm)) %>%
  mutate(lidar_snow_density = lidar_swe / lidar_hs) %>%
  mutate(swe_dif = lidar_swe - swe_mm) %>%
  mutate(hs_dif = lidar_hs - snow_depth_cm/100) %>%
  mutate(snow_dens_dif = lidar_snow_density - density)

# snow_off_pd = pd_parsed[pd_parsed$doy == 149, ] %>%
#   select("uid", "lidar_res", "lidar_pd") %>%
#   rename(snow_off_pd = lidar_pd)
# all_merge = merge(all_merge, snow_off_pd, all.x = TRUE, by=c('uid', 'lidar_res'))


# hs-hs and swe-swe plots
hs_merge %>%
  filter(interp_len == 2) %>%
  ggplot(., aes(x=snow_depth_cm /100, y=lidar_hs, shape=vegetation)) +
    facet_grid(doy ~ lidar_res) +
    geom_point() +
    scale_shape_manual(values=c(1, 16)) +
    geom_abline(intercept = 0, slope = 1, size=1) +
    labs(title="Lidar snow depth (HS) validation", x="manual HS (m)", y="lidar HS (m)")
ggsave(paste0(plot_out_dir, "hs_validation_intnum2.png"), width=p_width, height=p_height, dpi=dpi)

# ggplot(hs_merge, aes(x=snow_depth_cm /100, y=lidar_hs, color=as.factor(doy))) +
#   facet_grid(interp_len ~ lidar_res) +
#   geom_point() +
#   geom_abline(intercept = 0, slope = 1, size=1) +
#   labs(title="Lidar snow depth (HS) validation", x="manual HS (m)", y="lidar HS (m)")

hs_merge %>%
  filter(interp_len == 2) %>%
  ggplot(., aes(x=snow_depth_cm /100, y=lidar_hs*100/snow_depth_cm, shape=vegetation)) +
    facet_grid(doy ~ lidar_res) +
    geom_point() +
    scale_shape_manual(values=c(1, 16)) +
    ylim(0, 2) +
    geom_hline(yintercept=1, linetype='dashed') +
    labs(title="Lidar snow depth (HS) fractional validation", x="manual HS (m)", y="Lidar HS / manual HS (-)")
ggsave(paste0(plot_out_dir, "hs_fractional_validation_intnum2.png"), width=p_width, height=p_height, dpi=dpi)

swe_merge %>%
  filter(interp_len == 2) %>%
  ggplot(., aes(x=swe_mm, y=lidar_swe, shape=vegetation)) +
    facet_grid(doy ~ lidar_res) +
    geom_point() +
    scale_shape_manual(values=c(1, 16)) +
    geom_abline(intercept = 0, slope = 1, size=1) +
    labs(title="Lidar SWE validation", x="manual SWE (mm)", y="lidar SWE (mm)", color="Density model")
ggsave(paste0(plot_out_dir, "swe_validation_intnum2.png"), width=p_width, height=p_height, dpi=dpi)

swe_merge %>%
  filter(interp_len == 2) %>%
  ggplot(., aes(x=swe_mm, y=lidar_swe / swe_mm, shape=vegetation)) +
  facet_grid(doy ~ lidar_res) +
  geom_point() +
  scale_shape_manual(values=c(1, 16)) +
  geom_hline(yintercept=1, linetype='dashed') +
  labs(title="Lidar SWE fractional validation", x="manual SWE (mm)", y="lidar SWE / manual SWE (-)", color="Density model") +
  ylim(0, 2)
ggsave(paste0(plot_out_dir, "swe_fractional_validation_intnum2.png"), width=p_width, height=p_height, dpi=dpi)

ggplot(all_merge, aes(x=density, y=lidar_snow_density, shape=density_assumption)) +
  facet_grid(doy ~ lidar_res) +
  geom_point() +
  scale_shape_manual(values=c(1, 16)) +
  geom_abline(intercept = 0, slope = 1, size=1) +
  labs(title="Snow density validation", x="manual snow density (kg/m^3)", y="lidar snow density (kg/m^3)")

all_merge %>%
  filter(lidar_res == .25, interp_len == 2) %>%
  ggplot(., aes(x=lidar_pd, y=hs_dif, color=as.factor(vegetation))) +
    geom_point() +
    scale_x_continuous(trans='log10', breaks = c(100, 1000, 10000)) +
    labs(title="Lidar snow depth error with snow on ground point density", x="lidar snow depth - manual (m)", y="lidar snow on ground point density (pts/m^2)")

all_merge %>%
  filter(lidar_res == .1 | lidar_res == .25) %>%
  ggplot(., aes(x=hs_dif, y=(snow_off_pd + lidar_pd)/2, color=cover)) +
  facet_grid(doy ~ lidar_res) +
  geom_point() +
  geom_vline(xintercept=0) +
  scale_y_continuous(trans='log10', breaks = c(100, sqrt(10) * 100, 1000, sqrt(10) * 1000)) +
  labs(title="Lidar snow depth error with snow on ground point density", x="lidar snow depth - manual (m)", y="mean lidar ground point density (pts/m^2)")


# groups and error metrics
rmse = function(difdata){
  sqrt(sum(difdata^2, na.rm = TRUE))/sqrt(length(na.omit(difdata)))
}
# define rmse
mae = function(difdata){
  sum(abs(difdata), na.rm = TRUE)/length(na.omit(difdata))
}

hs_veg = hs_merge
# hs_veg$vegetation = "all"
# hs_veg = rbind(hs_veg, hs_merge)

hs_group_veg <- hs_veg %>%
  filter(interp_len == 2) %>%
  group_by(doy, lidar_res, vegetation) %>%
  summarise(hs_rmse=rmse(hs_dif), hs_mae=mae(hs_dif), hs_mb=mean(hs_dif, na.rm=TRUE), count=sum(!is.na(hs_dif)))

hs_group_veg_allday <- hs_veg %>%
  filter(interp_len == 2) %>%
  group_by(lidar_res, vegetation) %>%
  summarise(hs_rmse=rmse(hs_dif), hs_mae=mae(hs_dif), hs_mb=mean(hs_dif, na.rm=TRUE), count=sum(!is.na(hs_dif)), cq3d_mean=mean(CQ.3D, na.rm=TRUE))

swe_veg = swe_merge
# swe_veg$vegetation = "all"
# swe_veg = rbind(swe_veg, swe_merge)

swe_group_veg <- swe_veg %>%
  filter(interp_len == 2) %>%
  group_by(doy, lidar_res, density_assumption, vegetation) %>%
  summarise(swe_rmse=rmse(swe_dif), swe_mae=mae(swe_dif), swe_mb=mean(swe_dif, na.rm=TRUE), count=sum(!is.na(swe_dif)))

swe_group_veg_allday <- swe_veg %>%
  filter(interp_len == 2) %>%
  group_by(lidar_res, density_assumption, vegetation) %>%
  summarise(swe_rmse=rmse(swe_dif), swe_mae=mae(swe_dif), swe_mb=mean(swe_dif, na.rm=TRUE), count=sum(!is.na(swe_dif)), cq3d_mean=mean(CQ.3D, na.rm=TRUE))

# plot errors
ggplot(hs_group_veg, aes(x=lidar_res, y=hs_mb, shape=vegetation, linetype=vegetation)) +
  facet_grid(doy ~ .) +
  geom_point() +
  scale_shape_manual(values=c(1, 16)) +
  geom_line() +
  scale_linetype_manual(values=c("dashed", "solid")) +
  labs(title="Snow depth (HS) mean bias", x="lidar resolution (m)", y="HS mean bias (m)")
ggsave(paste0(plot_out_dir, "hs_mb_intnum2.png"), width=4, height=p_height, dpi=dpi)

ggplot(hs_group_veg, aes(x=lidar_res, y=hs_rmse, shape=vegetation, linetype=vegetation)) +
  facet_grid(doy ~ .) +
  geom_point() +
  scale_shape_manual(values=c(1, 16)) +
  geom_line() +
  scale_linetype_manual(values=c("dashed", "solid")) +
  labs(title="Snow depth (HS) RMSE", x="lidar resolution (m)", y="HS RMSE (m)")
ggsave(paste0(plot_out_dir, "hs_rmse_intnum2.png"), width=4, height=p_height, dpi=dpi)


ggplot(swe_group_veg, aes(x=lidar_res, y=swe_mb, shape=vegetation, linetype=vegetation)) +
  facet_grid(doy ~ .) +
  geom_point() +
  scale_shape_manual(values=c(1, 16)) +
  geom_line() +
  scale_linetype_manual(values=c("dashed", "solid")) +
  labs(title="SWE mean bias", x="lidar resolution (m)", y="SWE mean bias (mm)")
ggsave(paste0(plot_out_dir, "swe_mb_intnum2.png"), width=4, height=p_height, dpi=dpi)


ggplot(swe_group_veg, aes(x=lidar_res, y=swe_rmse, shape=vegetation, linetype=vegetation)) +
    facet_grid(doy ~ .) +
    geom_point() +
    scale_shape_manual(values=c(1, 16)) +
    geom_line() +
    scale_linetype_manual(values=c("dashed", "solid")) +
    labs(title="SWE RMSE", x="lidar resolution (m)", y="SWE RMSE (mm)")
ggsave(paste0(plot_out_dir, "swe_rmse_intnum2.png"), width=4, height=p_height, dpi=dpi)
