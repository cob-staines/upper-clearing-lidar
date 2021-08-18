import laspy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

las_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_snow_off\\OUTPUT_FILES\\LAS\\19_149_normalized_UF_above_2m.las"

# import las_in
inFile = laspy.file.File(las_in, mode="r")
# select dimensions
las_z = inFile.z
inFile.close()

# distribution of vegetation point heights in UF
bin_count, bins = np.histogram(las_z, bins=1000)
canopy_height = pd.DataFrame({'bin_low': bins[:-1],
                            'bin_high': bins[1:],
                            'bin_center': bins[:-1] + np.diff(bins)/2,
                            'count': bin_count,
                            'norm_dist': bin_count/np.sum(bin_count)})

# plot vegetation height distribution
plt.plot(canopy_height.bin_center, canopy_height.norm_dist)

# angle bins and weights
method = "licor"

if method == "licor":
    # licor lai-2000
    theta_bins = pd.DataFrame({'bin_low': np.array([0, 16, 32, 47, 61]) * np.pi/180,
                                'bin_high': np.array([13, 28, 43, 58, 74]) * np.pi/180,
                                'weight': [0.034, 0.104, 0.160, 0.218, 0.484]})
    theta_bins.loc[:, 'bin_center'] = theta_bins.bin_low + (theta_bins.bin_high - theta_bins.bin_low)/2
    theta_bins.loc[:, 'w_norm'] = theta_bins.weight / np.sum(theta_bins.weight)
elif method == "miller":
    n_bins = 6
    theta_min = 0 * np.pi/180
    theta_max = 90 * np.pi/180
    bin_edges = np.linspace(theta_min, theta_max, num=n_bins)
    theta_bins = pd.DataFrame({'bin_low': bin_edges[:-1],
                               'bin_high': bin_edges[1:],
                               'bin_center': bin_edges[:-1] + np.diff(bin_edges)/2})
    theta_bins.loc[:, 'weight'] = np.sin(theta_bins.bin_center) * np.diff(bin_edges)
    theta_bins.loc[:, 'w_norm'] = theta_bins.weight/np.sum(theta_bins.weight)
else:
    raise Exception('LAI method currently not accepted. Does it need to be added?')

n_layers = np.shape(canopy_height)[1]
n_theta = np.shape(theta_bins)[1]
inner = []
outer = []
ring_weight = []
for hh in range(0, n_theta):
    inner.extend(np.tan(theta_bins.bin_low[hh]) * canopy_height.bin_center)
    outer.extend(np.tan(theta_bins.bin_high[hh]) * canopy_height.bin_center)
    ring_weight.extend(theta_bins.w_norm[hh] * canopy_height.norm_dist)

band = pd.DataFrame({"inner": inner,
                     "outer": outer,
                     "ring_weight": ring_weight})
band.loc[:, "area"] = np.pi * (band.outer ** 2 - band.inner ** 2)
band.loc[:, "plot_weight"] = band.ring_weight/band.area

# resample 1d
step = 0.1
dist_max = 75

n_step = int(dist_max/step)
dist = np.full(n_step, 0)
area = np.full(n_step, 0)

edges = np.linspace(0, dist_max, num=n_step + 1)
dist = edges[:-1] + np.diff(edges)/2
area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
plot_w = np.full(n_step, 0.0)
for ii in range(0, n_step):
    plot_w[ii] = np.sum(band.plot_weight[(dist[ii] >= band.inner) & (dist[ii] < band.outer)])

footprint = pd.DataFrame({"dist": dist,
                          "ring_area": area,
                          "plot_weight": plot_w})
footprint.loc[:, "ring_weight"] = footprint.plot_weight * footprint.ring_area
footprint.loc[:, "plot_norm"] = footprint.plot_weight/np.sum(footprint.ring_weight)
footprint.loc[:, "ring_norm"] = footprint.ring_weight/np.sum(footprint.ring_weight)
footprint.loc[:, "cumulative_f"] = np.cumsum(footprint.ring_norm)
footprint.loc[:, "cumulative_b"] = np.cumsum(footprint.ring_norm[::-1])[::-1]

plt.plot(footprint.dist, footprint.plot_norm)
plt.plot(footprint.dist, footprint.ring_norm)
plt.plot(footprint.dist, footprint.cumulative_f)
plt.plot(footprint.dist, footprint.cumulative_b)

# resample 2d
d_min = -100
d_max = 100
step = 1
n_d = int((d_max - d_min)/step + 1)
fpim = np.full([n_d, n_d], 0.0)
dist = np.linspace(d_min, d_max, num=n_d)
for ii in range(0, n_d):
    for jj in range(0, n_d):
        dd = np.sqrt(dist[ii] ** 2 + dist[jj] ** 2)
        fpim[ii, jj] = np.sum(band.plot_weight[(dd >= band.inner) & (dd < band.outer)])
# normalize
fpim = fpim/np.sum(fpim)
plt.imshow(fpim, interpolation='nearest')

# overlap analysis
n_step = int(d_max/step) + 1
dist = np.linspace(0, d_max, num=n_step)
val = np.full(n_step, 0.0)
for ii in range(0, n_step):
    val[ii] = np.sum(np.sqrt(fpim[:, ii:] * fpim[:, :(n_d - ii)]))

overlap = pd.DataFrame({"dist": dist,
                        "footprint_overlap": val})

fig, ax = plt.subplots()
ax.plot(overlap.dist, overlap.footprint_overlap)
ax.set(xlabel='Distance between samples (m)', ylabel='Footprint overlap',
       title='Licor LAI footprint overlap with distance between samples')
ax.grid()
plt.show()
# model knife edge

# contact number field
dim = d_max*2 + 1
k_num = np.full_like(fpim, 0)
k_num[:, 0:int(dim/2)] = 6
plt.imshow(k_num, interpolation='nearest')
# overlap analysis
step = 1
n_min = d_min
n_max = d_max
n_min = -50
n_max = 50
n_step = int((n_max - n_min)/step + 1)
dist = np.linspace(n_min, n_max, num=n_step)
val = np.full(n_step, 0.0)
for ii in range(0, n_step):
    k_num = np.full_like(fpim, 0)
    k_num[:, 0:int(d_max - dist[ii])] = 1
    val[ii] = np.sum(fpim * k_num)

knife_edge = pd.DataFrame({"dist": dist,
                            "lai": val})
fig, ax = plt.subplots()
ax.plot(knife_edge.dist, knife_edge.lai)
ax.set(xlabel='Distance from step (m)', ylabel='LAI',
       title='Licor LAI across contact number (K) step of unit height')
ax.grid()
plt.show()

dlai = np.diff(knife_edge.lai)
dist_mid = knife_edge.dist[:-1] + step/2
plt.plot(dist_mid, dlai)

fig, ax = plt.subplots()
ax.plot(dist_mid, -dlai)
ax.set(xlabel='Distance from step (m)', ylabel='LAI',
       title='Change in Licor LAI across contact number (K) step of unit height')
ax.grid()
plt.show()