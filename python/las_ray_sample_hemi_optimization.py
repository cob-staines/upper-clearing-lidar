import las_ray_sampling as lrs
import numpy as np
import pandas as pd
import os

vox = lrs.VoxelObj()
vox.las_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\LAS\\19_149_las_proc_classified_merged.las'
# vox.las_in = 'C:\\Users\\jas600\\workzone\\data\\las\\19_149_las_proc_classified_merged.las'
vox.traj_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_all_traj.txt'
# vox.traj_in = 'C:\\Users\\jas600\\workzone\\data\\las\\19_149_all_traj.txt'
vox.return_set = 'first'
vox.drop_class = 7
vox.las_traj_hdf5 = vox.las_in.replace('.las', '_ray_sampling_' + vox.return_set + '_returns_drop_' + str(vox.drop_class) + '_las_traj.h5')
vox.sample_precision = np.uint32
vox.return_precision = np.uint32
vox.las_traj_chunksize = 10000000
vox.cw_rotation = -34 * np.pi / 180
voxel_length = .25
vox.step = np.full(3, voxel_length)
vox.sample_length = voxel_length/np.pi
vox.vox_hdf5 = vox.las_in.replace('.las', '_ray_sampling_' + vox.return_set + '_returns_drop_' + str(vox.drop_class) + '_r' + str(voxel_length) + 'm_vox.h5')


# vox = lrs.las_to_vox(vox, run_las_traj=False, fail_overflow=False)

# # LOAD VOX
vox = lrs.load_vox_meta(vox.vox_hdf5, load_data=True)


batch_dir = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\synthetic_hemis\\batches\\lrs_hemi_optimization_r.25_px100_beta\\'

img_lookup_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\hemispheres\\hemi_lookup_cleaned.csv"
# img_lookup_in = 'C:\\Users\\jas600\\workzone\\data\\las\\hemi_lookup_cleaned.csv'
max_quality = 4
las_day = "19_149"
# import hemi_lookup
img_lookup = pd.read_csv(img_lookup_in)
# filter lookup by quality
img_lookup = img_lookup[img_lookup.quality_code <= max_quality]
# filter lookup by las_day
img_lookup = img_lookup[img_lookup.folder == las_day]

[file.replace('.JPG', '') for file in img_lookup.filename]


pts = pd.DataFrame({'id': img_lookup.filename,
                    'x_utm11n': img_lookup.xcoordUTM1,
                    'y_utm11n': img_lookup.ycoordUTM1,
                    'z_m': img_lookup.elevation})

# batch_dir = 'C:\\Users\\jas600\\workzone\\data\\hemigen\\mb_15_1m_pr.15_os10\\'
# las_in = "C:\\Users\\jas600\\workzone\\data\\hemigen\\hemi_lookups\\19_149_las_proc_classified_merged.las"
# pts_in = 'C:\\Users\\jas600\\workzone\\data\\hemigen\\hemi_lookups\\1m_dem_points_mb_15.csv'

# # load points
# pts = pd.read_csv(pts_in)

rshmeta = lrs.RaySampleHemiMetaObj()
# self.id = None
# self.file_name = None

rshmeta.agg_method = 'beta'

if rshmeta.agg_method == 'nb_lookup':
    mean_path_length = 2 * np.pi / (6 + np.pi) * voxel_length  # mean path length through a voxel cube across angles (m)
    prior_weight = 5  # in units of scans (1 <=> equivalent weight to 1 expected voxel scan)
    prior_b = mean_path_length * prior_weight
    prior_a = prior_b * 0.01
    rshmeta.prior = [prior_a, prior_b]
elif rshmeta.agg_method == 'linear':
    samps = (vox.sample_data > 0)
    trans = vox.return_data[samps] // (vox.sample_data[samps] * vox.sample_length)
    rshmeta.prior = np.var(trans)
elif rshmeta.agg_method == 'beta':
    val = (vox.sample_data > 0)  # roughly 50% at .25m
    ret = vox.return_data[val]
    sam = vox.sample_data[val]
    # correct values to be between 0 and 1
    sam[ret > sam] = ret[ret > sam]
    mu = np.mean(ret / sam)
    sig2 = np.var(ret / sam)

    alpha = ((1 - mu)/sig2 - 1/mu) * (mu ** 2)
    beta = alpha * (1/mu - 1)
    rshmeta.prior = [alpha, beta]
else:
    raise Exception('Aggregation method ' + rshmeta.agg_method + ' unknown.')

# # beta model
# z = 0  # number of returns in given volume
# N = 21  # number of samples in given volume
#
# post_a = z + alpha
# post_b = N - z + beta
#
# normal approximation of sums
# mu_i = post_a/(post_a + post_b)
# S_mu = sum(mu_i)
# sig2_i = post_a* post_b/((post_a + post_b) **2 * (post_a + post_b + 1))
# S_sig2 = sum(sig2_i)

rshmeta.ray_sample_length = vox.sample_length
rshmeta.ray_iterations = 100  # model runs for each ray, from which median and std of returns is calculated

# image dimensions
#phi_step = (np.pi/2) / (180 * 2)
rshmeta.img_size = 100  # square, in pixels/ray samples
#rshmeta.max_phi_rad = phi_step * rshmeta.img_size
rshmeta.max_phi_rad = np.pi/2

# image geometry
hemi_m_above_ground = img_lookup.height_m  # meters
rshmeta.max_distance = 50  # meters
rshmeta.min_distance = voxel_length * np.sqrt(3)  # meters


# output file dir
rshmeta.file_dir = batch_dir + "outputs\\"

if not os.path.exists(rshmeta.file_dir):
    os.makedirs(rshmeta.file_dir)

rshmeta.id = pts.id
rshmeta.origin = np.array([pts.x_utm11n,
                           pts.y_utm11n,
                           pts.z_m + hemi_m_above_ground]).swapaxes(0, 1)

rshmeta.file_name = ["las_19_149_id_" + str(id) + ".tif" for id in pts.id]

rshm = lrs.rs_hemigen(rshmeta, vox, initial_index=0)
# rshm = lrs.rs_hemigen_multiproc(rshmeta, vox, initial_index=0, n_cores=4)


# parse results
import tifffile as tif

# contact number log
cnlog = rshm.copy()

angle_lookup = pd.read_csv(cnlog.file_dir[0] + "phi_theta_lookup.csv")
phi = np.full((cnlog.img_size_px[0], cnlog.img_size_px[0]), np.nan)
phi[(np.array(angle_lookup.x_index), np.array(angle_lookup.y_index))] = angle_lookup.phi * 180 / np.pi

phi_bands = [0, 15, 30, 45, 60, 75]

cnlog.loc[:, ["rsm_mean_1", "rsm_mean_2", "rsm_mean_3", "rsm_mean_4", "rsm_mean_5"]] = np.nan
cnlog.loc[:, ["rsm_med_1", "rsm_med_2", "rsm_med_3", "rsm_med_4", "rsm_med_5"]] = np.nan
cnlog.loc[:, ["rsm_std_1", "rsm_std_2", "rsm_std_3", "rsm_std_4", "rsm_std_5"]] = np.nan
for ii in range(0, len(cnlog)):
    img = tif.imread(cnlog.file_dir[ii] + cnlog.file_name[ii])
    mean = img[:, :, 0]
    med = img[:, :, 1]
    std = img[:, :, 2]
    mean_temp = []
    med_temp = []
    std_temp = []
    for jj in range(0, 5):
        mask = (phi >= phi_bands[jj]) & (phi < phi_bands[jj + 1])
        mean_temp.append(np.nanmean(mean[mask]))
        med_temp.append(np.nanmean(med[mask]))
        std_temp.append(np.nanmean(std[mask]))
    cnlog.loc[ii, ["rsm_mean_1", "rsm_mean_2", "rsm_mean_3", "rsm_mean_4", "rsm_mean_5"]] = mean_temp
    cnlog.loc[ii, ["rsm_med_1", "rsm_med_2", "rsm_med_3", "rsm_med_4", "rsm_med_5"]] = med_temp
    cnlog.loc[ii, ["rsm_std_1", "rsm_std_2", "rsm_std_3", "rsm_std_4", "rsm_std_5"]] = std_temp

cnlog.to_csv(cnlog.file_dir[0] + "contact_number_optimization.csv")

###
# #
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# import tifffile as tif
# ii = 0
# img = tif.imread(rshmeta.file_dir + rshmeta.file_name[ii])
# cn = img[:, :, 1] * 0.02268
# cv = img[:, :, 2] / img[:, :, 0]
#
#
#
# fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# ax1, ax2 = axs.ravel()
#
# im1 = ax1.imshow(cn, interpolation='nearest', cmap='Greys', norm=matplotlib.colors.LogNorm())
# ax1.set_title("Contact number")
# ax1.set_axis_off()
#
# im2 = ax2.imshow(cv, interpolation='nearest', cmap='Greys', norm=matplotlib.colors.LogNorm())
# ax2.set_title("Coefficient of variation")
# ax2.set_axis_off()
#
# ##
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#
# fig, ax = plt.subplots(figsize=(8, 8))
#
# # img = ax.imshow(cn, interpolation='nearest', cmap='Greys')
# img = ax.imshow(cn, interpolation='nearest', cmap='Greys', norm=matplotlib.colors.LogNorm())
# ax.set_title("Contact number over hemisphere, modeled by ray re-sampling")
#
# axins = inset_axes(ax,
#                    width="5%",  # width = 5% of parent_bbox width
#                    height="50%",  # height : 50%
#                    loc='right',
#                    bbox_to_anchor=(.1, 0, 1, 1),
#                    bbox_transform=ax.transAxes,
#                    borderpad=0,
#                    )
# fig.colorbar(img, cax=axins)
#
# ax.set_axis_off()
# fig.savefig(rshmeta.file_dir + 'contact_num_plot_' + rshmeta.file_name[ii] + '.png')

##

# modeling beam reflectance with gamma prior
# mean = k * theta
# var = k * theta ^2
# theta = var / mean (or cv...)
# k = mean / theta = mean^2 / var

# trans = np.sort(vox.return_data[vox.sample_data > 0] / vox.sample_data[vox.sample_data > 0])
# mm = np.mean(trans * vox.sample_length)
# vv = np.var(trans * vox.sample_length)
# theta = vv / mm
# kk = mm ** 2 / vv
# prior = (kk, theta)
#
# kk = 100  # returns
# nn = 100  # samples
#
# # post_a = kk ** 3/((nn ** 2) * prior[0] * prior[1] ** 2) + prior[0]
# # post_b = 1 / (kk ** 2 / (prior[0] * prior[1] ** 2 * nn) + 1 / prior[1])
#
# # fails for kk=0... should not!
#
# post_a = 1 / (nn * prior[0] * prior[1] ** 2 / kk ** 2 + 1 / prior[0])
# post_b = prior[0] * prior[1] ** 2 / kk + prior[1]
# peace = np.random.gamma(post_a, post_b, 10000)
# np.mean(peace)
# np.var(peace)
#
# q999 = np.quantile(trans, .999)
# # set ceiling on trans by adjusting samples...
#
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# plt.hist(trans, bins=500)

