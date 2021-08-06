import numpy as np
import pandas as pd
from tqdm import tqdm
import tifffile as tif
import rastools

# import matplotlib
# matplotlib.use('Qt5Agg')
# import matplotlib.pyplot as plt


def lrs_footprint_products(batch_dir, cn_coef, file_out=None, threshold=False):

    if file_out is True:
        if threshold:
            file_out = batch_dir + "outputs\\rshmetalog_footprint_products_threshold" + str(threshold) + ".csv"
        else:
            file_out = batch_dir + "outputs\\rshmetalog_footprint_products.csv"

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

    footprint_df = pd.DataFrame({
        "id": rshmeta.id,
        "lrs_cn_1": np.nan,
        "lrs_cn_2": np.nan,
        "lrs_cn_3": np.nan,
        "lrs_cn_4": np.nan,
        "lrs_cn_5": np.nan,
        "lrs_cn_1_deg": np.nan,
        "lrs_cn_75_deg": np.nan,
        "lrs_cn_90_deg": np.nan,
        "lrs_tx_1": np.nan,
        "lrs_tx_2": np.nan,
        "lrs_tx_3": np.nan,
        "lrs_tx_4": np.nan,
        "lrs_tx_5": np.nan,
        "lrs_tx_1_deg": np.nan,
        "lrs_tx_75_deg": np.nan,
        "lrs_tx_90_deg": np.nan,
        "lrs_lai_1_deg": np.nan,
        "lrs_lai_15_deg": np.nan,
        "lrs_lai_75_deg": np.nan,
        "lrs_lai_90_deg": np.nan,
        "lrs_lai_2000": np.nan,
        "lrs_sky_view": np.nan,
        "lrs_cc": np.nan
    })

    z_step = 5000
    z_count = np.ceil(len(rshmeta)/z_step).astype(int)


    print("Iterating through " + str(z_count) + " image chunks:")
    for zz in range(0, z_count):
        z_low = zz * z_step
        if zz != (z_count - 1):
            z_high = (zz + 1) * z_step
        else:
            z_high = len(rshmeta)

        z_num = z_high-z_low

        cn_stack = np.full([imsize, imsize, z_num], np.nan)
        cn_std_stack = np.full([imsize, imsize, z_num], np.nan)
        for ii in tqdm(range(0, z_num), desc="img chunk" + str(zz + 1), ncols=100, leave=True):
            # cn_stack[:, :, ii] = tif.imread(batch_dir + "outputs\\" + rshmeta.file_name[ii])[:, :, 0] * cn_coef
            im = tif.imread(batch_dir + "outputs\\" + rshmeta.file_name[zz * z_step + ii])
            cn_stack[:, :, ii] = im[:, :, 0] * cn_coef
            # cn_std_stack[:, :, ii] = im[:, :, 1] * cn_coef  # for error analysis

        cn_stack_long = cn_stack
        cn_stack_long = np.swapaxes(np.swapaxes(cn_stack_long, 1, 2), 0, 1).reshape(cn_stack_long.shape[2], -1)
        tx_stack_long = np.exp(-cn_stack_long)

        if threshold:
            tx_stack_long[tx_stack_long > threshold] = 1
            tx_stack_long[tx_stack_long <= threshold] = 0

        # cn_std_stack_long = cn_std_stack
        # cn_std_stack_long = np.swapaxes(np.swapaxes(cn_std_stack_long, 1, 2), 0, 1).reshape(cn_std_stack_long.shape[2], -1)
        # tx_std_stack_long = np.exp(-cn_std_stack_long)

        # summarize by angle band
        angle_step = 1  # in degrees
        phi_bin = np.floor(phi * 180 / (np.pi * angle_step)) * angle_step  # similar to ceil, except for int values
        bins = np.unique(phi_bin)
        bins = bins[~np.isnan(bins)]
        bin_count = len(bins)

        cn_bin_means = np.full([z_num, bin_count], np.nan)
        # cn_std_bin_means = np.full([z_num, bin_count], np.nan)
        tx_bin_means = np.full([z_num, bin_count], np.nan)
        # tx_std_bin_means = np.full([z_num, bin_count], np.nan)
        for ii in range(0, bin_count):
            bb = bins[ii]
            mask = (phi_bin == bb).reshape(phi.size)
            cn_bin_means[:, ii] = np.mean(cn_stack_long[:, mask], axis=1)
            # cn_std_bin_means[:, ii] = np.mean(cn_std_stack_long[:, mask], axis=1)
            tx_bin_means[:, ii] = np.mean(tx_stack_long[:, mask], axis=1)
            # tx_std_bin_means[:, ii] = np.mean(tx_std_stack_long[:, mask], axis=1)


        def angle_range_stat(lower_lim, upper_lim, var, weight_by="solid_angle"):
            valid_bins = (bins >= lower_lim) & (bins < upper_lim)

            if weight_by == "solid_angle":
                ang_rad = (bins[valid_bins] + (angle_step / 2)) * np.pi / 180
                weights = np.sin(ang_rad)  # solid angle weights
            elif weight_by == "angle":
                weights = np.full(np.sum(valid_bins), 1)  # equal angle weights
            elif weight_by == "cos":
                ang_rad = (bins[valid_bins] + (angle_step / 2)) * np.pi / 180
                weights = np.sin(ang_rad) * np.cos(ang_rad)  # solid angle weights
            elif weight_by == "sin":
                ang_rad = (bins[valid_bins] + (angle_step / 2)) * np.pi / 180
                weights = np.sin(ang_rad) * np.sin(ang_rad)  # solid angle weights

            output = np.average(var[:, valid_bins], weights=weights, axis=1)

            return output


        footprint_df.loc[z_low:z_high-1, "lrs_tx_1"] = angle_range_stat(0, 15, tx_bin_means)
        footprint_df.loc[z_low:z_high-1, "lrs_tx_2"] = angle_range_stat(15, 30, tx_bin_means)
        footprint_df.loc[z_low:z_high-1, "lrs_tx_3"] = angle_range_stat(30, 45, tx_bin_means)
        footprint_df.loc[z_low:z_high-1, "lrs_tx_4"] = angle_range_stat(45, 60, tx_bin_means)
        footprint_df.loc[z_low:z_high-1, "lrs_tx_5"] = angle_range_stat(60, 75, tx_bin_means)

        footprint_df.loc[z_low:z_high-1, "lrs_tx_1_deg"] = angle_range_stat(0, 1, tx_bin_means)
        footprint_df.loc[z_low:z_high-1, "lrs_tx_60_deg"] = angle_range_stat(0, 60, tx_bin_means)
        footprint_df.loc[z_low:z_high-1, "lrs_tx_75_deg"] = angle_range_stat(0, 75, tx_bin_means)
        footprint_df.loc[z_low:z_high-1, "lrs_tx_90_deg"] = angle_range_stat(0, 90, tx_bin_means)

        # # calculate contact number bands from transmittance bands
        if threshold:
            footprint_df.loc[z_low:z_high-1, "lrs_cn_1"] = -np.log(footprint_df.lrs_tx_1)
            footprint_df.loc[z_low:z_high-1, "lrs_cn_2"] = -np.log(footprint_df.lrs_tx_2)
            footprint_df.loc[z_low:z_high-1, "lrs_cn_3"] = -np.log(footprint_df.lrs_tx_3)
            footprint_df.loc[z_low:z_high-1, "lrs_cn_4"] = -np.log(footprint_df.lrs_tx_4)
            footprint_df.loc[z_low:z_high-1, "lrs_cn_5"] = -np.log(footprint_df.lrs_tx_5)

            # footprint_df.loc[z_low:z_high-1, "lrs_cn_1_deg"] = -np.log(footprint_df.lrs_tx_1_deg)
            footprint_df.loc[z_low:z_high-1, "lrs_cn_75_deg"] = -np.log(footprint_df.lrs_tx_75_deg)
            footprint_df.loc[z_low:z_high-1, "lrs_cn_90_deg"] = -np.log(footprint_df.lrs_tx_90_deg)
        else:
            footprint_df.loc[z_low:z_high-1, "lrs_cn_1"] = angle_range_stat(0, 15, cn_bin_means)
            footprint_df.loc[z_low:z_high-1, "lrs_cn_2"] = angle_range_stat(15, 30, cn_bin_means)
            footprint_df.loc[z_low:z_high-1, "lrs_cn_3"] = angle_range_stat(30, 45, cn_bin_means)
            footprint_df.loc[z_low:z_high-1, "lrs_cn_4"] = angle_range_stat(45, 60, cn_bin_means)
            footprint_df.loc[z_low:z_high-1, "lrs_cn_5"] = angle_range_stat(60, 75, cn_bin_means)

            footprint_df.loc[z_low:z_high-1, "lrs_cn_1_deg"] = angle_range_stat(0, 1, cn_bin_means)
            footprint_df.loc[z_low:z_high-1, "lrs_cn_60_deg"] = angle_range_stat(0, 60, cn_bin_means)
            footprint_df.loc[z_low:z_high-1, "lrs_cn_75_deg"] = angle_range_stat(0, 75, cn_bin_means)
            footprint_df.loc[z_low:z_high-1, "lrs_cn_90_deg"] = angle_range_stat(0, 90, cn_bin_means)

        # # calculate transmittance bands from cn bands
        # footprint_df.loc[z_low:z_high-1, "lrs_tx_1"] = np.exp(-footprint_df.lrs_cn_1)
        # footprint_df.loc[z_low:z_high-1, "lrs_tx_2"] = np.exp(-footprint_df.lrs_cn_2)
        # footprint_df.loc[z_low:z_high-1, "lrs_tx_3"] = np.exp(-footprint_df.lrs_cn_3)
        # footprint_df.loc[z_low:z_high-1, "lrs_tx_4"] = np.exp(-footprint_df.lrs_cn_4)
        # footprint_df.loc[z_low:z_high-1, "lrs_tx_5"] = np.exp(-footprint_df.lrs_cn_5)
        #
        # footprint_df.loc[z_low:z_high-1, "lrs_tx_1_deg"] = np.exp(-footprint_df.lrs_cn_1_deg)
        # footprint_df.loc[z_low:z_high-1, "lrs_tx_75_deg"] = np.exp(-footprint_df.lrs_cn_75_deg)
        # footprint_df.loc[z_low:z_high-1, "lrs_tx_90_deg"] = np.exp(-footprint_df.lrs_cn_90_deg)

        # lai vertical projection
        cos_proj = np.cos(bins * np.pi / 180)

        # calculate lai from binned cn
        footprint_df.loc[z_low:z_high - 1, "lrs_lai_1_deg"] = angle_range_stat(0, 1, 2 * cn_bin_means * cos_proj)
        footprint_df.loc[z_low:z_high - 1, "lrs_lai_15_deg"] = angle_range_stat(0, 15, 2 * cn_bin_means * cos_proj)
        footprint_df.loc[z_low:z_high - 1, "lrs_lai_60_deg"] = angle_range_stat(0, 60, 2 * cn_bin_means * cos_proj)
        footprint_df.loc[z_low:z_high - 1, "lrs_lai_75_deg"] = angle_range_stat(0, 75, 2 * cn_bin_means * cos_proj)
        footprint_df.loc[z_low:z_high - 1, "lrs_lai_90_deg"] = angle_range_stat(0, 90, 2 * cn_bin_means * cos_proj)

        # # calculate lai from binned transmittances
        # footprint_df.loc[z_low:z_high - 1, "lrs_lai_1_deg"] = angle_range_stat(0, 1, 2 * (-np.log(tx_bin_means)) * cos_proj)
        # footprint_df.loc[z_low:z_high - 1, "lrs_lai_15_deg"] = angle_range_stat(0, 15, 2 * (-np.log(tx_bin_means)) * cos_proj)
        # footprint_df.loc[z_low:z_high - 1, "lrs_lai_75_deg"] = angle_range_stat(0, 75, 2 * (-np.log(tx_bin_means)) * cos_proj)
        # footprint_df.loc[z_low:z_high - 1, "lrs_lai_90_deg"] = angle_range_stat(0, 90, 2 * (-np.log(tx_bin_means)) * cos_proj)

        # cos_mean_phi = np.cos((np.array([1, 2, 3, 4, 5]) * 15 - 15./2) * np.pi / 180)

        weighted_phi = np.full(5, np.nan)
        weighted_phi[0] = angle_range_stat(0, 15, bins[np.newaxis, :])
        weighted_phi[1] = angle_range_stat(15, 30, bins[np.newaxis, :])
        weighted_phi[2] = angle_range_stat(30, 45, bins[np.newaxis, :])
        weighted_phi[3] = angle_range_stat(45, 60, bins[np.newaxis, :])
        weighted_phi[4] = angle_range_stat(60, 75, bins[np.newaxis, :])

        cos_mean_phi = np.cos(weighted_phi * np.pi / 180)

        # footprint_df.loc[z_low:z_high - 1, "lrs_lai_2000"] = 2 * (0.034 * footprint_df.lrs_cn_1 * cos_mean_phi[0] +
        #                                                           0.104 * footprint_df.lrs_cn_2 * cos_mean_phi[1] +
        #                                                           0.160 * footprint_df.lrs_cn_3 * cos_mean_phi[2] +
        #                                                           0.218 * footprint_df.lrs_cn_4 * cos_mean_phi[3] +
        #                                                           0.484 * footprint_df.lrs_cn_5 * cos_mean_phi[4])



        footprint_df.loc[z_low:z_high - 1, "lrs_lai_2000"] = 2 * (0.034 * footprint_df.lrs_cn_1 * cos_mean_phi[0] +
                                                                  0.104 * footprint_df.lrs_cn_2 * cos_mean_phi[1] +
                                                                  0.160 * footprint_df.lrs_cn_3 * cos_mean_phi[2] +
                                                                  (0.218 + 0.484) * footprint_df.lrs_cn_4 * cos_mean_phi[3])  # dropping 5th ring

        footprint_df.loc[z_low:z_high - 1, "lrs_sky_view"] = angle_range_stat(0, 75, tx_bin_means, weight_by="cos")
        footprint_df.loc[z_low:z_high - 1, "lrs_cc"] = 1 - angle_range_stat(0, 75, tx_bin_means, weight_by="solid_angle")

    if file_out:
        footprint_df.to_csv(file_out, index=False)

    return footprint_df


#
# # # 19_149 -- snow_off
snow_off_dir = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\ray_sampling\\batches\\lrs_uf_r.25_px181_snow_off_dem_offset.25\\'
snow_off_coef = 0.38686933  # python tx drop 5
# snow_off_coef = 0.1921595  # optimized for cn dropping 5th
# # cn_coef = 0.220319  # optimized for cn
lrs_footprint_products(snow_off_dir, snow_off_coef, file_out=True)
threshold = 0.38438069
lrs_footprint_products(snow_off_dir, snow_off_coef, threshold=threshold, file_out=True)

#
#
# 045_050_052 -- snow_on
snow_on_dir = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\ray_sampling\\batches\\lrs_uf_r.25_px181_snow_on_dem_offset.25\\'
snow_on_coef = 0.37181197  # python tx drop 5
# snow_on_coef = 0.1364611  # optimized for cn dropping 5th
# cn_coef = 0.141832  # optimized for cn
lrs_footprint_products(snow_on_dir, snow_on_coef, file_out=True)
#
#
# # validation set at optimized coef
# batch_dir = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\ray_sampling\\batches\\lrs_hemi_optimization_r.25_px1000_snow_off\\"
# cn_coef = 1
# lrs_footprint_products(batch_dir, cn_coef, file_out=True)
# batch_dir = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\ray_sampling\\batches\\lrs_hemi_optimization_r.25_px181_snow_off_max150m\\"
# cn_coef = 1
# lrs_footprint_products(batch_dir, cn_coef, file_out=True)
# batch_dir = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\ray_sampling\\batches\\lrs_hemi_optimization_r.25_px181_snow_off\\"
# cn_coef = 1
# file_out = batch_dir + "outputs\\rshmetalog_footprint_products.csv"
# lrs_footprint_products(batch_dir, cn_coef, file_out=file_out)
# cn_coef = 0.38686933  # python tx drop 5
# file_out = batch_dir + "outputs\\rshmetalog_footprint_products_opt.csv"
# lrs_footprint_products(batch_dir, cn_coef, file_out=file_out)
# opt_thresh = 0.4713883769260814  # tx mean of opt cn_coef
# file_out = batch_dir + "outputs\\rshmetalog_footprint_products_opt_thresh.csv"
# lrs_footprint_products(batch_dir, cn_coef, file_out=file_out, threshold=opt_thresh)
#
# batch_dir = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\ray_sampling\\batches\\lrs_hemi_optimization_r.25_px1000_snow_on\\"
# cn_coef = 1
# lrs_footprint_products(batch_dir, cn_coef, file_out=True)
# batch_dir = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\ray_sampling\\batches\\lrs_hemi_optimization_r.25_px181_snow_on_max150m\\"
# cn_coef = 1
# lrs_footprint_products(batch_dir, cn_coef, file_out=True)
# batch_dir = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\ray_sampling\\batches\\lrs_hemi_optimization_r.25_px181_snow_on\\"
# cn_coef = 1
# lrs_footprint_products(batch_dir, cn_coef, file_out=True)
# cn_coef = 0.37181197  # python tx drop 5
# file_out = batch_dir + "outputs\\rshmetalog_footprint_products_opt.csv"
# lrs_footprint_products(batch_dir, cn_coef, file_out=file_out)
# opt_thresh = 0.6915388816825417
# file_out = batch_dir + "outputs\\rshmetalog_footprint_products_opt_thresh.csv"
# lrs_footprint_products(batch_dir, cn_coef, file_out=file_out, threshold=opt_thresh)

# optimization
from scipy.optimize import minimize, fmin_bfgs, dual_annealing


photos_lai_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\hemispheres\\19_149\\clean\\sized\\thresholded\\LAI_parsed.dat"
# photos_lai_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\hemispheres\\045_052_050\\LAI_045_050_052_parsed.dat"
photos_lai = pd.read_csv(photos_lai_in)
photos_lai.loc[:, "original_file"] = photos_lai.picture.str.replace("_r.jpg", ".JPG", case=False).str.replace(".jpg", ".JPG", case=False)
photos_meta_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\hemispheres\\hemi_lookup_cleaned.csv"
photos_meta = pd.read_csv(photos_meta_in)
#
# photos_meta$id <- as.numeric(rownames(photos_meta)) - 1

photos = photos_lai.merge(photos_meta, left_on='original_file', right_on='filename')
photos_strip = photos.loc[:, ["original_file", "transmission_s_1", "transmission_s_2", "transmission_s_3", "transmission_s_4", "transmission_s_5"]]
photos = pd.melt(photos_strip, id_vars="original_file", value_name="transmission_s")
photos.loc[:, "ringnum"] = photos.variable.str.replace("transmission_s_", "").astype(int)
photos.loc[:, "solid_angle"] = 2 * np.pi * (np.cos((photos.ringnum - 1) * 15 * np.pi / 180) - np.cos(photos.ringnum * 15 * np.pi / 180))


def tx_opt_coef(xx, mean_tx=False):
    threshold = xx[0]
    # cn_coef = xx[0]
    cn_coef = 0.38686933  # snow off

    batch_dir = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\ray_sampling\\batches\\lrs_hemi_optimization_r.25_px181_snow_off\\"
    # batch_dir = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\ray_sampling\\batches\\lrs_hemi_optimization_r.25_px181_snow_on\\"
    # cn_coef = .42
    # threshold = 0.5
    # fpp = lrs_footprint_products(batch_dir, cn_coef, file_out=False, threshold=threshold)

    def run_footprint_analysis(threshold=threshold):
        fpp = lrs_footprint_products(batch_dir, cn_coef, file_out=False, threshold=threshold)
        fpp_tx = fpp.loc[:, ["id", "lrs_tx_1", "lrs_tx_2", "lrs_tx_3", "lrs_tx_4", "lrs_tx_5"]]
        # fpp_cn = fpp.loc[:, ["id", "lrs_cn_1", "lrs_cn_2", "lrs_cn_3", "lrs_cn_4", "lrs_cn_5"]]

        lrs_tx = pd.melt(fpp_tx, id_vars="id", value_name="lrs_tx")
        lrs_tx.loc[:, "ringnum"] = lrs_tx.variable.str.replace("lrs_tx_", "").astype(int)
        tx = lrs_tx.merge(photos, left_on=["id", "ringnum"], right_on=["original_file", "ringnum"])
        tx.loc[:, "tx_error"] = tx.lrs_tx - tx.transmission_s

        tx_drop = tx.loc[tx.ringnum != 5, :]

        # lrs_cn = pd.melt(fpp_cn, id_vars="id", value_name="lrs_cn")
        # lrs_cn.loc[:, "ringnum"] = lrs_cn.variable.str.replace("lrs_cn_", "").astype(int)
        # cn = lrs_cn.merge(photos, left_on=["id", "ringnum"], right_on=["original_file", "ringnum"])
        # cn.loc[:, "cn_error"] = cn.lrs_cn - -np.log(cn.transmission_s)

        return tx_drop

    # calculate error stats

    def wrmse(difdata, weights):
        weights = weights / np.nansum(weights)
        return np.sqrt(np.nansum(weights * (difdata ** 2)))

    def wmb(difdata, weights):
        weights = weights / np.nansum(weights)
        return np.nansum(weights * difdata)

    tx_drop = run_footprint_analysis()

    t_mean = wmb(tx_drop.lrs_tx, tx_drop.solid_angle)

    # tx_drop_thresh = run_footprint_analysis(threshold=t_mean)

    # wmb(tx.tx_error, tx.solid_angle)
    # wmb(cn.cn_error, cn.solid_angle)
    # wrmse(cn.cn_error, cn.solid_angle)

    if mean_tx:
        return [wrmse(tx_drop.tx_error, tx_drop.solid_angle), t_mean]
    else:
        return wrmse(tx_drop.tx_error, tx_drop.solid_angle)

res = minimize(tx_opt_coef, x0=np.array([0.5]))

#
# def callbackF(xi):
#     print('{0: 3.6f}'.format(xi[0]))
#
# print('{0:9s}'.format('X0'))
#
#
# w0 = np.array([0.5])
# [popt, fopt, gopt, Bopt, func_calls, grad_calls, warnflg] = \
#     fmin_bfgs(tx_opt_coef, w0, callback=callbackF, maxiter=50, full_output=True, retall=False)


# tx_wrmse, tx_mean = tx_opt_coef(popt, mean_tx=True)

res = dual_annealing(tx_opt_coef, bounds=[(0.1, 0.6)], x0=np.array([0.38]), maxiter=50)

#
# # create raster products
# df = pd.read_csv(file_out)
# point_ids_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\synthetic_hemis\\hemi_grid_points\\mb_65_r.25m\\dem_r.25_point_ids.tif"
# colnames = [
#     "lrs_cn_1",
#     "lrs_cn_2",
#     "lrs_cn_3",
#     "lrs_cn_4",
#     "lrs_cn_5",
#     "lrs_lai_2000"]
#
# for cc in colnames:
#     ras_out = batch_dir + cc + ".tif"
#     rastools.pd_to_raster(df, cc, point_ids_in, ras_out)
#
#
#
# # create raster products
# file_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\synthetic_hemis\\batches\\mb_15_1m_pr.15_os10\\outputs\\LAI_parsed.dat"
# df = pd.read_csv(file_in)
# point_ids_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\synthetic_hemis\\hemi_grid_points\\mb_65_1m\\1m_dem_point_ids.tif"
# colnames = [
#     "contactnum_1",
#     "contactnum_2",
#     "contactnum_3",
#     "contactnum_4",
#     "contactnum_5",
#     "lai_no_cor"]
#
# for cc in colnames:
#     ras_out = file_in.replace("LAI_parsed.dat", "") + cc + ".tif"
#     rastools.pd_to_raster(df, cc, point_ids_in, ras_out)
