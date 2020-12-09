import rastools
import numpy as np
import pandas as pd
import tifffile as tif

batch_dir = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\ray_sampling\\batches\\lrs_hemi_uf_.25m_180px\\outputs\\'
# batch_dir = 'C:\\Users\\jas600\\workzone\\data\\hemigen\\mb_15_1m_pr.15_os10\\outputs\\'

log_batch = True

if log_batch:
    covar_out = batch_dir + "phi_theta_lookup_log_covar_training.csv"
    weighted_cn_out = batch_dir + "rshmetalog_weighted_cn.csv"
else:
    covar_out = batch_dir + "phi_theta_lookup_covar.csv"

# covar type
globalBool = False
localBool = True

scaling_coef = 0.19546

# load img meta
hemimeta = pd.read_csv(batch_dir + 'rshmetalog.csv')

imsize = hemimeta.img_size_px[0]

# merge with covariant
var_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\mb_65\\dSWE\\19_045-19_052\\dswe_19_045-19_052_r.25m.tif'
var = rastools.raster_to_pd(var_in, 'covariant')
hemi_var = pd.merge(hemimeta, var, left_on=('x_utm11n', 'y_utm11n'), right_on=('x_coord', 'y_coord'), how='inner')

# load angle template
angle_lookup = pd.read_csv(batch_dir + "phi_theta_lookup.csv")
phi = np.full((imsize, imsize), np.nan)
phi[(np.array(angle_lookup.x_index), np.array(angle_lookup.y_index))] = angle_lookup.phi * 180 / np.pi
max_phi = 90


# filter to desired images
#hemiList = hemi_swe.loc[(hemi_swe.swe.values >= 0) & (hemi_swe.swe.values <= 150), :]

# delineate training set and test set
set_param = np.random.random(len(hemi_var))
hemi_var.loc[:, 'training_set'] = set_param < 1
hemiList = hemi_var.loc[hemi_var.training_set, :].reset_index()



imstack = np.full([imsize, imsize, len(hemiList)], np.nan)
if log_batch:
    for ii in range(0, len(hemiList)):
        imstack[:, :, ii] = np.log(tif.imread(batch_dir + hemiList.file_name[ii])[:, :, 1] * scaling_coef)
        print(str(ii + 1) + ' of ' + str(len(hemiList)))
else:
    for ii in range(0, len(hemiList)):
        imstack[:, :, ii] = tif.imread(batch_dir + hemiList.file_name[ii])[:, :, 1] * scaling_coef
        print(str(ii + 1) + ' of ' + str(len(hemiList)))


# calculate radius to avoid pixels outside of circle
# im_center = (imsize - 1)/2
imrange = np.full((imsize, imsize), False)
imrange[phi <= max_phi] = True
# for ii in range(0, imsize):
#         for jj in range(0, imsize):
#             rr = np.sqrt((im_center - ii) ** 2 + (im_center - jj) ** 2)
#             if rr <= imsize / 2:
#                 imrange[jj, ii] = True

if globalBool:
    globalCovar = np.full((imsize, imsize), np.nan)
    global_mean_tray = np.full((imsize, imsize), np.nan)
    for ii in range(0, imsize):
        for jj in range(0, imsize):
            if imrange[jj, ii]:
                global_mean_tray[jj, ii] = np.mean(imstack[jj, ii, :])
        print(ii)
    global_mean = np.nanmean(global_mean_tray)
    #np.save(batch_dir + '_global_mean_tray', global_mean_tray)

if localBool:
    localCovar = np.full((imsize, imsize), np.nan)

var_mu = np.mean(hemiList.covariant)
for ii in range(0, imsize):
    for jj in range(0, imsize):
        if imrange[jj, ii]:
            can = imstack[jj, ii, :]
            can_mu = np.mean(can)

            if localBool:
                localCovar[jj, ii] = np.mean((can - np.mean(can)) * (hemiList.covariant - var_mu))
            if globalBool:
                globalCovar[jj, ii] = np.mean((can - global_mean) * (hemiList.covariant - var_mu))
    print(ii)
    # save arrays to file every row, in event of crash
    # np.save(batch_dir + '_local', localCovar)
    # np.save(batch_dir + '_global', globalCovar)

# localCovar = np.load(batch_dir + '_local.npy')
# globalCovar = np.load(batch_dir + '_global.npy')

angle_lookup_covar = angle_lookup.copy()
if log_batch:
    angle_lookup_covar.loc[:, 'log_covar'] = localCovar[angle_lookup_covar.y_index.values, angle_lookup_covar.x_index.values]
else:
    angle_lookup_covar.loc[:, 'linear_covar'] = localCovar[angle_lookup_covar.y_index.values, angle_lookup_covar.x_index.values]

angle_lookup_covar.loc[:, "abs_log_covar"] = np.abs(angle_lookup_covar.log_covar)
angle_lookup_covar.loc[:, "abs_log_covar_weight"] = angle_lookup_covar.log_covar / np.sum(angle_lookup_covar.loc[:, "abs_log_covar"])
angle_lookup_covar.loc[:, "sqr_log_covar"] = angle_lookup_covar.log_covar ** 2
angle_lookup_covar.loc[:, "sqr_log_covar_weight"] = angle_lookup_covar.log_covar / np.sum(angle_lookup_covar.loc[:, "sqr_log_covar"])

angle_lookup_covar.to_csv(covar_out, index=False)

# Calculate weighted CN for all images based on hemisphere

abs_weight = np.full([imsize, imsize], np.nan)
abs_weight[(angle_lookup_covar.y_index.values, angle_lookup_covar.x_index.values)] = angle_lookup_covar.abs_log_covar_weight.values

sqr_weight = np.full([imsize, imsize], np.nan)
sqr_weight[(angle_lookup_covar.y_index.values, angle_lookup_covar.x_index.values)] = angle_lookup_covar.sqr_log_covar_weight.values

hemi_var.loc[:, "log_cn_abs_weighted"] = np.nan
hemi_var.loc[:, "log_cn_sqr_weighted"] = np.nan
if log_batch:
    for ii in range(0, len(hemi_var)):
        temp_im = np.log(tif.imread(batch_dir + hemi_var.file_name[ii])[:, :, 1] * scaling_coef)
        hemi_var.log_cn_abs_weighted[ii] = np.nansum(-1 * temp_im * abs_weight)
        hemi_var.log_cn_sqr_weighted[ii] = np.nansum(-1 * temp_im * sqr_weight)
        print(str(ii + 1) + ' of ' + str(len(hemi_var)))

hemi_var.to_csv(weighted_cn_out, index=False)

import matplotlib
matplotlib.use('TkAgg')
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

figout = batch_dir + 'covar_plot.png'


fig = plt.figure()
a = fig.subplots()
# imgplot = plt.imshow(localCovar, cmap=plt.get_cmap('cividis_r'))
imgplot = plt.imshow(localCovar, cmap=plt.get_cmap('Purples_r'))
plt.axis('off')
plt.colorbar()


## visualize weights

plt.imshow(abs_weight, cmap=plt.get_cmap('Purples_r'))
plt.imshow(sqr_weight, cmap=plt.get_cmap('Purples_r'))


# load data
data_in ='C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\ray_sampling\\batches\\lrs_hemi_uf_.25m_180px\\outputs\\rshmetalog_weighted_cn.csv'
hemi_var = pd.read_csv(data_in)


# all points
plt.scatter(hemi_var.covariant, hemi_var.log_cn_weighted)
# training set
plt.scatter(hemi_var.covariant[hemi_var.training_set.values], -hemi_var.log_cn_abs_weighted[hemi_var.training_set.values], s=1, alpha=.25)
# test set
plt.scatter(hemi_var.covariant[~hemi_var.training_set.values], -hemi_var.log_cn_abs_weighted[~hemi_var.training_set.values], s=1, alpha=.25)

plt.scatter(hemi_var.covariant[~hemi_var.training_set.values], -hemi_var.log_cn_sqr_weighted[~hemi_var.training_set.values], s=1, alpha=.25)


##
#
# fig = plt.figure()
# a = fig.add_subplot(1, 2, 1)
# imgplot = plt.imshow(localCovar)
# a.set_title('Covarience (local) of SWE (0-150mm) and Canopy presence')
# plt.colorbar()
# a = fig.add_subplot(1, 2, 2)
# imgplot = plt.imshow(globalCovar)
# a.set_title('Covarience (global) of SWE (0-150mm) and Canopy presence')
# plt.colorbar()
# fig.savefig(figout)




#####

# h5py takes way too long. attempting to wrestle in memory instead
#
# import h5py
# hdf5_path = batch_dir + 'imstack_t' + str(threshold) + '_faulty.h5'
#
# with h5py.File(hdf5_path, 'w') as hf:
#     hf.create_dataset('imstack', imstack.shape, data=imstack, dtype='bool', chunks=(1, 1, 25285))

# with h5py.File(hdf5_path, 'r') as hf:
#     mean_global = np.mean(hf['imstack'][1, 1, selection])

# takes 5+ days to calculate mean. Not going to work. Can I do this in working memory instead?
# if meantype == 'global':
#     global_mean_tray = np.full((imsize, imsize), np.nan)
#     for ii in range(0, imsize):
#         #for jj in range(0, imsize):
#         jj = 500
#         rr = np.sqrt((im_center - ii) ** 2 + (im_center - jj) ** 2)
#         if rr <= imsize / 2:
#             with h5py.File(hdf5_path, 'r') as hf:
#                 global_mean_tray[jj, ii] = np.mean(hf['imstack'][jj, ii, selection])
#         print(ii)
#     global_mean = np.nanmean(global_mean_tray)
covar_global = covar
covar_local = covar
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.imshow(covar, interpolation='nearest')
plt.colorbar

fig = plt.figure()
a = fig.add_subplot(1, 2, 1)
imgplot = plt.imshow(covar_local)
a.set_title('Covarience (local) of SWE (40-50mm) and Canopy presence')
plt.colorbar()
a = fig.add_subplot(1, 2, 2)
imgplot = plt.imshow(covar_global)
a.set_title('Covarience (global) of SWE (40-50mm) and Canopy presence')
plt.colorbar()