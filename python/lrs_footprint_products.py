import numpy as np
import pandas as pd
from tqdm import tqdm
import tifffile as tif

# import matplotlib
# matplotlib.use('Qt5Agg')
# import matplotlib.pyplot as plt

batch_dir = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\ray_sampling\\batches\\lrs_uf_r.25_px181_snow_on\\'
file_out = batch_dir + "outputs\\rshmetalog_footprint_products.csv"

# cn_coef = 0.195878  # 19_149
cn_coef = 0.132154  # 045_050_052
# cn_coef = 1  # optimization

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
    "lrs_sky_view": np.nan
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
        cn_std_stack[:, :, ii] = im[:, :, 1] * cn_coef

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

    cn_bin_means = np.full([z_num, bin_count], np.nan)
    cn_std_bin_means = np.full([z_num, bin_count], np.nan)
    tx_bin_means = np.full([z_num, bin_count], np.nan)
    tx_std_bin_means = np.full([z_num, bin_count], np.nan)
    for ii in range(0, bin_count):
        bb = bins[ii]
        mask = (phi_bin == bb).reshape(phi.size)
        cn_bin_means[:, ii] = np.mean(cn_stack_long[:, mask], axis=1)
        cn_std_bin_means[:, ii] = np.mean(cn_std_stack_long[:, mask], axis=1)
        tx_bin_means[:, ii] = np.mean(tx_stack_long[:, mask], axis=1)
        tx_std_bin_means[:, ii] = np.mean(tx_std_stack_long[:, mask], axis=1)


    def angle_range_stat(lower_lim, upper_lim, var, weight_by="solid_angle"):
        valid_bins = (bins >= lower_lim) & (bins < upper_lim)

        if weight_by == "solid_angle":
            ang_rad = (bins[valid_bins] + (angle_step / 2)) * np.pi / 180
            weights = np.sin(ang_rad)  # solid angle weights
        elif weight_by == "angle":
            weights = np.full(np.sum(valid_bins), 1)  # equal angle weights
        elif weight_by == "sky_view":
            ang_rad = (bins[valid_bins] + (angle_step / 2)) * np.pi / 180
            weights = np.sin(ang_rad) * np.cos(ang_rad)  # solid angle weights

        output = np.average(var[:, valid_bins], weights=weights, axis=1)

        return output

    footprint_df.loc[z_low:z_high-1, "lrs_cn_1"] = angle_range_stat(0, 15, cn_bin_means)
    footprint_df.loc[z_low:z_high-1, "lrs_cn_2"] = angle_range_stat(15, 30, cn_bin_means)
    footprint_df.loc[z_low:z_high-1, "lrs_cn_3"] = angle_range_stat(30, 45, cn_bin_means)
    footprint_df.loc[z_low:z_high-1, "lrs_cn_4"] = angle_range_stat(45, 60, cn_bin_means)
    footprint_df.loc[z_low:z_high-1, "lrs_cn_5"] = angle_range_stat(60, 75, cn_bin_means)

    footprint_df.loc[z_low:z_high-1, "lrs_cn_1_deg"] = angle_range_stat(0, 1, cn_bin_means)
    footprint_df.loc[z_low:z_high-1, "lrs_cn_75_deg"] = angle_range_stat(0, 75, cn_bin_means)
    footprint_df.loc[z_low:z_high-1, "lrs_cn_90_deg"] = angle_range_stat(0, 90, cn_bin_means)

    footprint_df.loc[z_low:z_high-1, "lrs_tx_1"] = angle_range_stat(0, 15, tx_bin_means)
    footprint_df.loc[z_low:z_high-1, "lrs_tx_2"] = angle_range_stat(15, 30, tx_bin_means)
    footprint_df.loc[z_low:z_high-1, "lrs_tx_3"] = angle_range_stat(30, 45, tx_bin_means)
    footprint_df.loc[z_low:z_high-1, "lrs_tx_4"] = angle_range_stat(45, 60, tx_bin_means)
    footprint_df.loc[z_low:z_high-1, "lrs_tx_5"] = angle_range_stat(60, 75, tx_bin_means)

    footprint_df.loc[z_low:z_high-1, "lrs_tx_1_deg"] = angle_range_stat(0, 1, tx_bin_means)
    footprint_df.loc[z_low:z_high-1, "lrs_tx_75_deg"] = angle_range_stat(0, 75, tx_bin_means)
    footprint_df.loc[z_low:z_high-1, "lrs_tx_90_deg"] = angle_range_stat(0, 90, tx_bin_means)

    footprint_df.loc[z_low:z_high - 1, "lrs_sky_view"] = angle_range_stat(0, 90, tx_bin_means, weight_by="sky_view")

footprint_df.to_csv(file_out, index=False)


# create raster LAI and CC products
import numpy as np
import rastools
import pandas as pd

lai_parsed = pd.read_csv(file_out)
lai_parsed.loc[:, 'canopy_closure'] = 1 - lai_parsed.openness

point_raster_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\synthetic_hemis\\hemi_grid_points\\mb_65_1m\\1m_dem_point_ids.tif"
point_raster = rastools.raster_load(point_raster_in)
lai_ras = rastools.raster_load(point_raster_in)
cc_ras = rastools.raster_load(point_raster_in)

lai_ras.data = np.full((lai_ras.rows, lai_ras.cols), lai_ras.no_data)
cc_ras.data = np.full((cc_ras.rows, cc_ras.cols), cc_ras.no_data)
for ii in range(0, len(lai_parsed)):
    lai_ras.data[np.where(point_raster.data == lai_parsed.id[ii])] = lai_parsed.lai_s_cc[ii]
    cc_ras.data[np.where(point_raster.data == lai_parsed.id[ii])] = lai_parsed.canopy_closure[ii]

lai_out = file_out.replace('LAI_parsed.dat', 'lai_ras.tif')
rastools.raster_save(lai_ras, lai_out)

cc_out = file_out.replace('LAI_parsed.dat', 'cc_ras.tif')
rastools.raster_save(cc_ras, cc_out)