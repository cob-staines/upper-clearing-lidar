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

# save to file
lai_pts_out = "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/site_library/upper-forest_autocorrelation_sample_points_elev_lai.csv"
write.csv2(lai_pts, lai_pts_out,row.names=FALSE)

# calculate cross-data
cross_data_2d <- function(id, x, y, val){
  len = length(id)
  dtn = (len^2 - len)/2
  dt <- data.table(id_id=rep("",dtn), dist=rep(0,dtn), val_dif=rep(0,dtn))
  nn = 1
  for(ii in 1:(len-1)) {
    for(jj in (ii+1):len){
      dt[nn,id_id := paste0(as.character(id[ii]), "_", as.character(id[jj]))]
      dt[nn,dist := sqrt((x[ii] - x[jj])^2 + (y[ii] - y[jj])^2)]
      dt[nn,val_dif := (val[ii] - val[jj])]
      nn = nn + 1
    }
  }
  return(dt)
}
semivariogram_2d <- function(dt, nbins){
  run = max(dt$dist) - min(dt$dist)
  step = run/nbins
  bins = ((0:nbins))*step + min(dt$dist)
  svg = data.table(bin_center=bins[1:nbins] + step/2, n=rep(0, nbins), var=rep(0, nbins))
  bins[1] = bins[1] - 0.01  # include lowest observation
  for (ii in 1:nbins){
    in_bin = (dt$dist > bins[ii]) & (dt$dist <= bins[ii + 1])
    n_ii = sum(in_bin)
    var_ii = sum(dt$val_dif[in_bin]^2)/(2*n)
    svg[ii,n := n_ii]
    svg[ii,var := var_ii]
  } 
  return(svg)
}

dt = cross_data_2d(lai_pts$id, lai_pts$x_utm11n, lai_pts$y_utm11n, lai_pts$Miller)
svg = semivariogram_2d(dt, 30)

# plots
ggplot(svg, aes(x=bin_center, y=var, color=n)) +
  geom_point() +
  labs(title="Semivariogram of LAI for Upper Forest (snow-free vegetation)", x="Distance (m)", y="LAI variance (-)", color="Samples per bin")

hist(dt$dist, breaks=30)

ggplot(dt, aes(x=dist, y=val_dif)) +
  geom_point()

