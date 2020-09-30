import rastools
import os
import numpy as np
from PIL import Image
import pandas as pd

batch_dir = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_snow_off\\OUTPUT_FILES\\synthetic_hemis\\uf_1m_pr_.15_os_10\\outputs\\'
imsize = 1000
# load img meta
hemimeta = pd.read_csv(batch_dir + 'hemimetalog.csv')

# merge with swe
swe_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\SWE\\19_045\\swe_19_045_r1.00m_q0.25.tif'
swe = rastools.raster_to_pd(swe_in, 'swe')
hemi_swe = pd.merge(hemimeta, swe, left_on=('x_utm11n', 'y_utm11n'), right_on=('x_coord', 'y_coord'), how='left')

# stack binary canopy data
imstack = np.full([imsize, imsize, len(hemi_swe)], False)
for ii in range(0, len(hemimeta)):
    imstack[:, :, ii] = np.array(Image.open(batch_dir + hemi_swe.file_name[ii]))[:, :, 0] > 127
    print(ii)

covar = np.full((imsize, imsize), np.nan)

swe_mu = np.mean(hemi_swe.swe)
for ii in range(0, imsize):
    for jj in range(0, imsize):
        can = imstack[ii, jj, :]
        can_mu = np.mean(can)
        covar[ii, jj] = np.mean((can - can_mu) * (hemi_swe.swe - swe_mu))
    print(ii)

import imageio
img = np.rint(template * 255).astype(np.uint8)
img_out = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\RSM\\ray_sampling_transmittance_20426_af' + str(area_factor) + '.png'
imageio.imsave(img_out, img)

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.imshow(covar, interpolation='nearest')