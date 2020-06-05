library('dplyr')
library('tidyr')
library('ggplot2')
library('data.table')

# import points
pts_in = "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/site_library/upper-forest_autocorrelation_sample_points_elev.csv"
pts = read.csv(pts_in, header=TRUE, na.strings = c("NA",""), sep=",")

# import lai
lai_in = "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/synthetic_hemis/autocorrelation/uf/white_bg/LAI.dat"
lai = read.csv(lai_in, header=TRUE, na.strings = c("NA",""), sep="")
colnames(lai)[1] = "picture_file"
lai$Miller = as.character(lai$Miller)
lai$id = -9999
for (ii in 1:nrow(lai)){
  lai$id[ii] = as.numeric(gsub(".png", "", gsub("las_19_149_pnt_", "", as.character(lai$picture_file[ii]))))
  lai$Miller[ii] = gsub(",", ".", lai$Miller[ii])
}
lai$Miller = as.numeric(lai$Miller)
lai = lai[c("Miller", "id")]

# merge points with lai
lai_pts = merge(pts, lai, by=c("id"))

# set bounds
xmin = min(lai_pts$x_utm11n)
xmax = max(lai_pts$x_utm11n)
ymin = min(lai_pts$y_utm11n)
ymax = max(lai_pts$y_utm11n)

#pick random sample points in bounds
step_l = 2
n_step = 17
samp_n = 1000
samp_x = runif(samp_n, xmin, xmax)
samp_y = runif(samp_n, ymin, ymax)

# for each sample point
ii = 1
for (ii in 1:samp_n){
  # preallocate
  sd_dist <- data.frame(point = numeric(n_step),
                        dist = numeric(n_step),
                        count = numeric(n_step),
                        sd = numeric(n_step))
  sd_dist$point = ii
  xx = samp_x[ii]
  yy = samp_y[ii]
  ss = 1
  dist_from_samp = sqrt((xx - lai_pts$x_utm11n)^2 + (yy - lai_pts$y_utm11n)^2)
  for (ss in 1:n_step){
    ll = ss*step_l
    set = lai_pts$Miller[dist_from_samp <= ll]
    sd_dist$dist[ss] = ll
    sd_dist$count[ss] = length(set)
    sd_dist$sd[ss] = sd(set)
  }
  if (ii==1){
    set_sd_dist = sd_dist
  } else {
    set_sd_dist = rbind(set_sd_dist, sd_dist)
  }
}

lai_summary = set_sd_dist %>%
  group_by(dist) %>%
  summarise(mean_sd = mean(sd, na.rm=TRUE), n_samples = length(na.omit(sd)), n_total=sum(count, na.rm=TRUE))

ggplot(lai_summary, aes(x=dist, y=mean_sd)) +
  geom_point() +
  xlim(0, 35) +
  ylim(0, 0.35) +
  labs(title="Standard deviation of LAI with distance for Upper Forest (snow-free vegetation)", x="Distance (m)", y="LAI standard deviation")

