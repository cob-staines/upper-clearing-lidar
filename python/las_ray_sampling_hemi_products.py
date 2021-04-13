import numpy as np
import pandas as pd
import tifffile as tif

# contact number log
# cnlog = rshm.copy()

batch_dir = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\ray_sampling\\batches\\lrs_hemi_uf_.25m_180px\\"
rshmlog_in = batch_dir + 'outputs\\rshmetalog.csv'
rshmlog = pd.read_csv(batch_dir + 'outputs\\rshmetalog.csv')

angle_lookup = pd.read_csv(batch_dir + "outputs\\phi_theta_lookup.csv")
img_size = rshmlog.img_size_px[0]
phi = np.full((img_size, img_size), np.nan)
phi[(np.array(angle_lookup.x_index), np.array(angle_lookup.y_index))] = angle_lookup.phi * 180 / np.pi  # in degrees

phi_bands = [0, 15, 30, 45, 60, 75]

cnlog.loc[:, ["rsm_mean_1", "rsm_mean_2", "rsm_mean_3", "rsm_mean_4", "rsm_mean_5"]] = np.nan
cnlog.loc[:, ["rsm_std_1", "rsm_std_2", "rsm_std_3", "rsm_std_4", "rsm_std_5"]] = np.nan
for ii in range(0, len(cnlog)):
    img = tif.imread(cnlog.file_dir[ii] + cnlog.file_name[ii])
    mean = img[:, :, 0]
    std = img[:, :, 1]
    mean_temp = []
    std_temp = []
    for jj in range(0, 5):
        mask = (phi >= phi_bands[jj]) & (phi < phi_bands[jj + 1])
        mean_temp.append(np.nanmean(mean[mask]))
        std_temp.append(np.nanmean(std[mask]))
    cnlog.loc[ii, ["rsm_mean_1", "rsm_mean_2", "rsm_mean_3", "rsm_mean_4", "rsm_mean_5"]] = mean_temp
    cnlog.loc[ii, ["rsm_std_1", "rsm_std_2", "rsm_std_3", "rsm_std_4", "rsm_std_5"]] = std_temp

cnlog.to_csv(cnlog.file_dir[0] + "contact_number_optimization.csv")

