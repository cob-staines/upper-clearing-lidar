import numpy as np
import pandas as pd
import tifffile as tif

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

batch_dir = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\ray_sampling\\batches\\lrs_hemi_optimization_r.25_px181_beta_single_ray_agg_045_050_052\\'
file_out = batch_dir + "outputs\\rshmetalog_products.csv"

# cn_coef = 0.195878  # 19_149
#cn_coef = 0.132154  # 045_050_052
cn_coef = 1  # optimization

rshmeta = pd.read_csv(batch_dir + "outputs\\rshmetalog.csv")
imsize = rshmeta.img_size_px[0]  # assuming all images have same dimensions
im_count = len(rshmeta)

# load image pixel angle lookup
angle_lookup = pd.read_csv(batch_dir + "outputs\\phi_theta_lookup.csv")
# build phi image (in radians)
phi = np.full((imsize, imsize), np.nan)
phi[(np.array(angle_lookup.y_index), np.array(angle_lookup.x_index))] = angle_lookup.phi
# build theta image (in radians)
theta = np.full((imsize, imsize), np.nan)
theta[(np.array(angle_lookup.y_index), np.array(angle_lookup.x_index))] = angle_lookup.theta


cn_stack = np.full([imsize, imsize, len(rshmeta)], np.nan)
cn_std_stack = np.full([imsize, imsize, len(rshmeta)], np.nan)
for ii in range(0, len(rshmeta)):
    # cn_stack[:, :, ii] = tif.imread(batch_dir + "outputs\\" + rshmeta.file_name[ii])[:, :, 0] * cn_coef
    im = tif.imread(batch_dir + "outputs\\" + rshmeta.file_name[ii])
    cn_stack[:, :, ii] = im[:, :, 0] * cn_coef
    cn_std_stack[:, :, ii] = im[:, :, 1] * cn_coef
    print(str(ii + 1) + ' of ' + str(len(rshmeta)))

cn_stack_long = cn_stack
cn_stack_long = np.swapaxes(np.swapaxes(cn_stack_long, 1, 2), 0, 1).reshape(cn_stack_long.shape[2], -1)
tx_stack_long = np.exp(-cn_stack_long)

cn_std_stack_long = cn_std_stack
cn_std_stack_long = np.swapaxes(np.swapaxes(cn_std_stack_long, 1, 2), 0, 1).reshape(cn_std_stack_long.shape[2], -1)
tx_std_stack_long = np.exp(-cn_std_stack_long)

# summarize by angle band
angle_step = 1  # in degrees
phi_bin = np.floor(phi * 180 / (np.pi * angle_step)) * angle_step  # similar to ceil, except for int values
bins = np.unique(phi_bin)
bins = bins[~np.isnan(bins)]
bin_count = len(bins)

cn_bin_means = np.full([im_count, bin_count], np.nan)
cn_std_bin_means = np.full([im_count, bin_count], np.nan)
tx_bin_means = np.full([im_count, bin_count], np.nan)
tx_std_bin_means = np.full([im_count, bin_count], np.nan)
for ii in range(0, bin_count):
    bb = bins[ii]
    mask = (phi_bin == bb).reshape(phi.size)
    cn_bin_means[:, ii] = np.mean(cn_stack_long[:, mask], axis=1)
    cn_std_bin_means[:, ii] = np.mean(cn_std_stack_long[:, mask], axis=1)
    tx_bin_means[:, ii] = np.mean(tx_stack_long[:, mask], axis=1)
    tx_std_bin_means[:, ii] = np.mean(tx_std_stack_long[:, mask], axis=1)


def angle_range_stat(lower_lim, upper_lim, var, solid_angle=True):
    valid_bins = (bins >= lower_lim) & (bins < upper_lim)

    if solid_angle:
        weights = np.sin((bins[valid_bins] + (angle_step / 2)) * np.pi / 180)  # solid angle weights
    else:
        weights = np.full(np.sum(valid_bins), 1)  # equal angle weights

    output = np.average(var[:, valid_bins], weights=weights, axis=1)

    return output

cn_1 = angle_range_stat(0, 15, cn_bin_means)
cn_2 = angle_range_stat(15, 30, cn_bin_means)
cn_3 = angle_range_stat(30, 45, cn_bin_means)
cn_4 = angle_range_stat(45, 60, cn_bin_means)
cn_5 = angle_range_stat(60, 75, cn_bin_means)

cn_1_deg = angle_range_stat(0, 1, cn_bin_means)
cn_75_deg = angle_range_stat(0, 75, cn_bin_means)
cn_90_deg = angle_range_stat(0, 90, cn_bin_means)

tx_1 = angle_range_stat(0, 15, tx_bin_means)
tx_2 = angle_range_stat(15, 30, tx_bin_means)
tx_3 = angle_range_stat(30, 45, tx_bin_means)
tx_4 = angle_range_stat(45, 60, tx_bin_means)
tx_5 = angle_range_stat(60, 75, tx_bin_means)

tx_1_deg = angle_range_stat(0, 1, tx_bin_means)
tx_75_deg = angle_range_stat(0, 75, tx_bin_means)
tx_90_deg = angle_range_stat(0, 90, tx_bin_means)





rshmeta_out = rshmeta
rshmeta_out = rshmeta_out.assign(tx_1=tx_1,
                                 tx_2=tx_2,
                                 tx_3=tx_3,
                                 tx_4=tx_4,
                                 tx_5=tx_5,
                                 tx_1_deg=tx_1_deg,
                                 tx_75_deg=tx_75_deg,
                                 tx_90_deg=tx_90_deg,
                                 cn_1=cn_1,
                                 cn_2=cn_2,
                                 cn_3=cn_3,
                                 cn_4=cn_4,
                                 cn_5=cn_5,
                                 cn_1_deg=cn_1_deg,
                                 cn_75_deg=cn_75_deg,
                                 cn_90_deg=cn_90_deg)

rshmeta_out.to_csv(file_out, index=False)
