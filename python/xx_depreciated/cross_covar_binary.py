import rastools
import os
import numpy as np
from PIL import Image
import pandas as pd

# batch_dir = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_snow_off\\OUTPUT_FILES\\synthetic_hemis\\uf_1m_pr_.15_os_10\\outputs\\'
batch_dir = 'C:\\Users\\jas600\\workzone\\data\\hemigen\\mb_15_1m_pr.15_os10\\outputs\\'
imsize = 1000
globalBool = True
localBool = True

# load img meta
hemimeta = pd.read_csv(batch_dir + 'hemimetalog.csv')

# merge with swe
swe_in = 'C:\\Users\\jas600\\workzone\\data\\cross_covar\\swe_19_045_r1.00m_q0.25.tif'
swe = rastools.raster_to_pd(swe_in, 'swe')
hemi_swe = pd.merge(hemimeta, swe, left_on=('x_utm11n', 'y_utm11n'), right_on=('x_coord', 'y_coord'), how='inner')

# filter to desited images
hemiList = hemi_swe.loc[(hemi_swe.swe.values >= 0) & (hemi_swe.swe.values <= 150), :]

# covar type

# stack binary canopy data
threshold = 128

imstack = np.full([imsize, imsize, len(hemiList)], False)
for ii in range(0, len(hemiList)):
    imstack[:, :, ii] = np.array(Image.open(batch_dir + hemimeta.file_name[ii]))[:, :, 0] < threshold
    print(str(ii + 1) + ' of ' + str(len(hemiList)))

im_center = (imsize - 1)/2
imrange = np.full((imsize, imsize), False)
for ii in range(0, imsize):
        for jj in range(0, imsize):
            rr = np.sqrt((im_center - ii) ** 2 + (im_center - jj) ** 2)
            if rr <= imsize / 2:
                imrange[jj, ii] = True

if globalBool:
    globalCovar = np.full((imsize, imsize), np.nan)
    global_mean_tray = np.full((imsize, imsize), np.nan)
    for ii in range(0, imsize):
        for jj in range(0, imsize):
            if imrange[jj, ii]:
                global_mean_tray[jj, ii] = np.mean(imstack[jj, ii, :])
        print(ii)
    global_mean = np.nanmean(global_mean_tray)
    np.save(batch_dir + '_global_mean_tray', global_mean_tray)

if localBool:
    localCovar = np.full((imsize, imsize), np.nan)

swe_mu = np.mean(hemiList.swe)
for ii in range(0, imsize):
    for jj in range(0, imsize):
        if imrange[jj, ii]:
            can = imstack[jj, ii, :]
            can_mu = np.mean(can)
            if localBool:
                localCovar[jj, ii] = np.mean((can - np.mean(can)) * (hemiList.swe - swe_mu))
            if globalBool:
                globalCovar[jj, ii] = np.mean((can - global_mean) * (hemiList.swe - swe_mu))
    print(ii)
    # save arrays to file every row, in event of crash
    np.save(batch_dir + '_local', localCovar)
    np.save(batch_dir + '_global', globalCovar)

localCovar = np.load(batch_dir + '_local.npy')
globalCovar = np.load(batch_dir + '_global.npy')

import matplotlib
matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt

figout = batch_dir + 'covar_plot.png'

fig = plt.figure()
a = fig.add_subplot(1, 2, 1)
imgplot = plt.imshow(localCovar)
a.set_title('Covarience (local) of SWE (0-150mm) and Canopy presence')
plt.colorbar()
a = fig.add_subplot(1, 2, 2)
imgplot = plt.imshow(globalCovar)
a.set_title('Covarience (global) of SWE (0-150mm) and Canopy presence')
plt.colorbar()
fig.savefig(figout)



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