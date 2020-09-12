# toolkit for geographical analysis

# histograms
import laslib
import numpy as np
import pandas as pd

def sample_semivar(pts_1, pts_2, dist_inv_func, n_samples, n_iters=1, self_ref=False):
    # select a set of points at random
    samps = np.random.randint(0, high=pts.__len__(), size=n_samples)

    # sample target distances according to distribution
    unif_samps = np.random.random(n_samples * n_iters)
    targ_dist = dist_inv_func(unif_samps)

    # preallocate distance and difference vectors
    true_dist = np.full(n_samples * n_iters, np.nan)
    dz = np.full(n_samples * n_iters, np.nan)

    for ii in range(0, n_samples):
        # for each sample calculate distances to all points
        dd = np.sqrt((pts_2.x - pts_1.x[samps[ii]]) ** 2 + (pts_2.y - pts_1.y[samps[ii]]) ** 2)
        for jj in range(0, n_iters):
            # for each iteration, find the point with distance closest to the corresponding target distance
            idx = (np.abs(dd - targ_dist[ii * n_samples + jj])).argmin()
            # record the actual distance between points
            true_dist[ii * n_samples + jj] = dd[idx]
            # record value difference
            dz[ii * n_samples + jj] = pts_2.z[idx] - pts_1.z[samps[ii]]
        print(ii)

    df = pd.DataFrame({'dist': true_dist,
                       'dz': dz})
    if not self_ref:
        # remove self-referencing values
        df = df[df.dist != 0]

    rang = (min(unif_samps), max(unif_samps))

    return df, rang

def bin_summarize(df, dist_inv_func, rang, bin_count):
    x = df.iloc[:, 0]
    y = df.iloc[:, 1]

    # calculate bins according to dist_inv_func
    bins = dist_inv_func(np.linspace(rang[0], rang[1], bin_count))

    bin_mid = (bins[0:-1] + bins[1:]) / 2
    # bin df according to a
    groups = y.groupby(np.digitize(x, bins))
    stats = pd.DataFrame({'bin_mid': bin_mid,
                          'n': groups.count(),
                          'mean_bias': groups.mean(),
                          'stdev': groups.std()})
    return stats

# semivar (self referencing error with distance)
las_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_snow_off\\OUTPUT_FILES\\LAS\\19_149_UF.las'
pts = laslib.las_xyz_load(las_in, keep_class=2)
pts = pd.DataFrame(data=pts,
                   columns=['x', 'y', 'z'])

def log10_inv(dd):
    return -np.log10(1 - dd)

df, rang = sample_semivar(pts, pts, log10_inv, 100, 100)
stats = bin_summarize(df, log10_inv, rang, 25)

# plot
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.scatter(stats.bin_mid, stats.std)
plt.ylim(0, .3)
plt.xlim(0, 1.5)
plt.show()
