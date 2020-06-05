library('ggplot2')
#install.packages('lidR')
library(lidR)

uf_norm <- readLAS("C:/Users/Cob/index/educational/usask/research/masters/data/lidar/19_149/19_149_snow_off/OUTPUT_FILES/LAS/19_149_normalized_UF_above_2m.las")
z_samp = hist(uf_norm$Z[uf_norm$Classification == 5], breaks = 100, main="Height distribution of vegetation at Upper Forest", xlab="height (m)")

height = z_samp$mids
weight = z_samp$counts
w_norm = weight/sum(weight)
h_dist_bins = data.frame(height, w_norm)

# licor lai-2000
bin_s = c(0, 16, 32, 47, 61)
bin_e = c(13, 28, 43, 58, 74)
w = c(0.034, 0.104, 0.160, 0.218, 0.484)
w_norm = w/sum(w)
theta_bins = data.frame(bin_s, bin_e, w_norm)

# uniform
#n_theta = 6
#theta_min = 0
#theta_max = 90
#theta_step = (theta_max - theta_min)/n_theta

bin_edges = seq(from = theta_min, to = theta_max, by = theta_step)
bin_s = bin_edges[1:n_theta-1]
bin_e = bin_edges[2:n_theta]
bin_mid = bin_s + theta_step/2
d_theta = bin_e - bin_s
w = sin(bin_mid*pi/180)*d_theta*pi/180
w_norm = w/sum(w)
theta_bins = data.frame(bin_s, bin_e, w_norm)

# for each horizontal layer
n_layers = nrow(h_dist_bins)
n_theta = nrow(theta_bins)

inner = numeric(n_layers*n_theta)
outer = numeric(n_layers*n_theta)
w_area = numeric(n_layers*n_theta)
for (ll in 1:n_layers){
  ll_inner = tan(theta_bins$bin_s*pi/180)*h_dist_bins$height[ll]
  ll_outer = tan(theta_bins$bin_e*pi/180)*h_dist_bins$height[ll]
  ll_area = pi*(ll_outer^2 - ll_inner^2)
  ll_w_area = theta_bins$w_norm*h_dist_bins$w_norm[ll]/ll_area
  
  # write
  a = ((ll - 1)*n_theta + 1)
  b = ll*n_theta
  inner[a:b] = ll_inner
  outer[a:b] = ll_outer
  w_area[a:b] = ll_w_area
}


# resample
step = .1
n_step = 2000
footprint = data.frame(dist = numeric(n_step), ring_w = numeric(n_step), plot_w = numeric(n_step))
for (ss in 1:n_step){
  bin_cen = ss*step - step/2
  footprint$dist[ss] = bin_cen
  # considering each consecutive ring
  r_area = pi*((ss*step)^2 - ((ss-1)*step)^2)
  rings = bin_cen > inner & bin_cen < outer
  footprint$ring_w[ss] = sum(w_area[rings]) * r_area
  footprint$plot_w[ss] = sum(w_area[rings])
}

footprint$ring_norm = footprint$ring_w/sum(footprint$ring_w)
footprint$plot_norm = footprint$plot_w/sum(footprint$plot_w)
footprint$cumsum_f = cumsum(footprint$ring_norm)
footprint$cumsum_b = rev(cumsum(rev(footprint$ring_norm)))

step = 1
n_steps = 75*2

# this could be done much more efficiently with image convolution

overlap = data.frame(dist = numeric(n_steps), val = numeric(n_steps))

for (ss in 0:n_steps){
  x_a = 0
  x_b = step*ss
  xmin = -75
  xmax = x_b + 75
  ymin = -75
  ymax = 75
  cell_list = NA
  zz = 0
  
  for (ii in xmin:xmax){
    for (jj in ymin:ymax){
      zz = zz + 1
      dist_a = sqrt(ii^2 + (jj + .5)^2)
      dist_b = sqrt((ii - x_b)^2 + (jj + .5)^2)
      a_val = footprint$plot_norm[footprint$dist == max(footprint$dist[footprint$dist < dist_a])]
      b_val = footprint$plot_norm[footprint$dist == max(footprint$dist[footprint$dist < dist_b])]
      cell_list[zz] = sqrt(a_val*b_val)
    }
  }
  overlap$dist[ss] = x_b
  overlap$val[ss] = sum(cell_list)
}

# not great... results are dominated by rounding error. But assuming rounding error is proportional to results, lets just use it.
# remove "far-out" noise
overlap$val = overlap$val - overlap$val[n_steps]

#re-normalize
overlap$val = overlap$val/overlap$val[1]

# relative contribution of equal-area plots located distance "dist" from point
ggplot(footprint, aes(x=dist, y=plot_norm)) +
  geom_point() +
  geom_line() +
  labs(title="Relative contribution to LAI (Licor) from equal-area plots with distance from center point", x="Distance (m)", y="Relative contribution")

# relative contribution of concentric rings of radius "dist" from point
ggplot(footprint, aes(x=dist, y=ring_norm)) +
  geom_point() +
  geom_line() +
  labs(title="Relative contribution to LAI (Licor) from concentric rings around center point", x="Ring radius (m)", y="Relative contribution")

# cumulative contribution from rings with raius less that "dist"
ggplot(footprint, aes(x=dist, y=cumsum_f)) +
  geom_point() +
  labs(title="Cumulative contribution to LAI (Licor) from concentric rings within radius of center point", x="Radius (m)", y="Cumulative contribution")

# footprint overlap of two points with distance from one another
ggplot(overlap, aes(x=dist, y=val)) +
  geom_point() +
  labs(title="Footprint overlap of two LAI (licor) values with distance", x="Distance (m)", y="Footprint overlap")
