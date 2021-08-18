import numpy as np
import pandas as pd
import tifffile as tif
import rastools


# objective: determine a weight for cn values as a function of phi and theta

covar_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\ray_sampling\\batches\\lrs_hemi_uf_.25m_180px\\outputs\\phi_theta_lookup_exp_covar_training.csv'
covar = pd.read_csv(covar_in)

# drop nans
covar = covar.loc[~np.isnan(covar.covar), :].copy()

# several methods for setting weights

# hist, bins = np.histogram(covar, bins=[0,20,40,60,80,100])

covar.loc[:, 'deg'] = np.digitize(covar.phi, np.arange(0, np.pi / 2, np.pi / 180)) - 1

# angle threshold
phi_weight = covar.groupby(['deg']).median().reset_index(drop=False)

phi_weight.loc[:, 'circum'] = np.pi * phi_weight.phi ** 2

phi_weight.loc[:, 'area_weight'] = phi_weight.covar * phi_weight.circum
phi_weight.area_weight = phi_weight.area_weight / np.sum(phi_weight.area_weight)
phi_weight.loc[:, 'area_weight_cum'] = np.cumsum(phi_weight.area_weight)

phi_weight.loc[:, 'covar_75_norm'] = phi_weight.covar / np.sum(phi_weight.covar)
phi_weight.loc[:, 'covar_15_norm'] = 0
phi_weight.loc[phi_weight.deg <= 15, 'covar_15_norm'] = phi_weight.covar[phi_weight.deg <= 15] / np.sum(phi_weight.covar[phi_weight.deg <= 15])


########
batch_dir = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\ray_sampling\\batches\\lrs_mb_15_dem_.25m_61px_mp15.25\\outputs\\'
# batch_dir = 'C:\\Users\\jas600\\workzone\\data\\hemigen\\mb_15_1m_pr.15_os10\\outputs\\'

scaling_coef = 0.19546

angle_lookup = pd.read_csv(batch_dir + "phi_theta_lookup.csv")
metalog = pd.read_csv(batch_dir + "rsgmetalog.csv")
metalog.loc[:, 'phi_deg'] = metalog.phi * 180 / np.pi

metalog.loc[:, 'weight'] = phi_weight.covar_15_norm[metalog.phi_deg.astype(int)].values


template = rastools.raster_load(batch_dir + metalog.file_name[0])
template.data = template.data[0]
template.data[template.data == template.no_data] = np.nan

lncnw = template.data.copy()
lncnw[:, :] = np.nan

for ii in range(0, len(metalog)):
    temp = tif.imread(batch_dir + metalog.file_name[ii])[:, :, 1]
    temp[temp == -9999] = np.nan
    temp = temp * scaling_coef * metalog.weight[ii]

    lncnw = np.nansum(np.concatenate((lncnw[:, :, np.newaxis], temp[:, :, np.newaxis]), axis=2), axis=2)
    print(str(ii + 1) + ' of ' + str(len(metalog)))

# clean up nans
lncnw[np.isnan(template.data)] = np.nan

# export to file
template.data = lncnw
template.band_count = 1
rastools.raster_save(template, batch_dir + 'weighted_cn.tif')


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

plt.imshow(lncnw, interpolation='nearest')

plt.scatter(phi_weight.phi, phi_weight.area_weight_cum)
plt.scatter(phi_weight.phi, phi_weight.area_weight)

plt.scatter(phi_weight.phi, -phi_weight.covar)
plt.scatter(phi_weight.phi, np.sin(2 * phi_weight.phi)/(np.pi * phi_weight.phi ** 2)/10)


plt.scatter(phi_weight.phi, 25 * np.max(phi_weight.area_weight) * np.sin(2*phi_weight.phi) / (phi_weight.phi ** 2))
plt.scatter(phi_weight.phi, 1 / np.sqrt(phi_weight.phi ** 2 + 1))
plt.scatter(phi_weight.phi, 7 * np.cos(phi_weight.phi) ** 12)
plt.scatter(phi_weight.deg, phi_weight.covar_15_norm)

plt.scatter(covar.phi, covar.covar)

plt.scatter(phi_weight.phi, -phi_weight.covar)
plt.scatter(phi_weight.phi, 7 * np.exp(-(phi_weight.phi * 4) ** 2/3))

