def merge_data_products(ddict, file_out, merge_data_in=None, left_on=None, right_on=None, mode='nearest'):
    import pandas as pd
    import rastools
    import numpy as np

    data = rastools.pd_sample_raster_gdal(ddict, include_nans=False, mode=mode)

    if 'dce' in ddict.keys():
        data.dce = np.rint(data.dce * 10)/10

    if merge_data_in is not None:
        # nest in list if not already
        if isinstance(merge_data_in, str):
            merge_data_in = [merge_data_in]
        if isinstance(right_on, str):
            right_on = [right_on]
        if isinstance(left_on, str):
            left_on = [left_on]



        for ii in range(len(merge_data_in)):
            # merge with hemisfer outputs
            print('Merging with hemi_data... ', end='')
            sup_data = pd.read_csv(merge_data_in[ii])
            data = data.merge(sup_data, how='left', left_on=left_on[ii], right_on=right_on[ii])
            print('done')

        # data = data.drop(columns='id')


    # save to file
    print('saving to file... ', end='')
    data.to_csv(file_out, index=False)
    print('done')

    return data


# first item in ddict is parent by default
# subsequent items are sampled at non-null parent cell centers

# 25cm products over mb_15
ddict = {
    'mb_15': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\synthetic_hemis\\hemi_grid_points\\mb_65_1m\\mb_15_plot_r.25m.tif',
    'plots': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\synthetic_hemis\\hemi_grid_points\\mb_65_r.25m\\uc_plot_r.25m.tif',
    ('er_p0_mean', 'er_p0_sd'): 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\ray_sampling\\batches\\lrs_mb_15_dem_.25m_61px_mp15.25\\outputs\\las_19_149_rs_mb_15_r.25_p0.0000_t3.1416.tif',
    'hemi_id': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\synthetic_hemis\\hemi_grid_points\\mb_65_1m\\1m_dem_point_ids.tif'
}

merge_data_in = ['C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\synthetic_hemis\\batches\\mb_15_1m_pr.15_os10\\outputs\\LAI_parsed.dat']
left_on = ['hemi_id']
right_on = ['id']
file_out = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\merged_data_products\\merged_mb_15_r.25m_canopy_19_149.csv'
data = merge_data_products(ddict, file_out, merge_data_in=merge_data_in, left_on=left_on, right_on=right_on)

# 25cm products and median swe/dswe over upper clearing
ddict = {
    'uc': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\synthetic_hemis\\hemi_grid_points\\mb_65_r.25m\\uc_plot_r.25m.tif',
    'swe_clin_19_045': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_045\\19_045_las_proc\\OUTPUT_FILES\\SWE\\clin\\interp_2x\\masked\\swe_clin_19_045_r.05m_interp2x_masked.tif',
    'swe_clin_19_050': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_050\\19_050_las_proc\\OUTPUT_FILES\\SWE\\clin\\interp_2x\\masked\\swe_clin_19_050_r.05m_interp2x_masked.tif',
    'swe_clin_19_052': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_052\\19_052_las_proc\\OUTPUT_FILES\\SWE\\clin\\interp_2x\\masked\\swe_clin_19_052_r.05m_interp2x_masked.tif',
    'dswe_cnsd_19_045-19_050': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\mb_65\\dSWE\\cnsd\\interp_2x\\19_045-19_050\\masked\\dswe_cnsd_19_045-19_050_r.05m_interp2x_masked.tif',
    'dswe_cnsd_19_050-19_052': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\mb_65\\dSWE\\cnsd\\interp_2x\\19_050-19_052\\masked\\dswe_cnsd_19_050-19_052_r.05m_interp2x_masked.tif',
    ('er_p0_mean', 'er_p0_sd'): 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\ray_sampling\\batches\\lrs_mb_15_dem_.25m_61px_mp15.25\\outputs\\las_19_149_rs_mb_15_r.25_p0.0000_t3.1416.tif',
    'hemi_id': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\synthetic_hemis\\hemi_grid_points\\mb_65_1m\\1m_dem_point_ids.tif'
}

merge_data_in = ['C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\synthetic_hemis\\batches\\mb_15_1m_pr.15_os10\\outputs\\LAI_parsed.dat']
left_on = ['hemi_id']
right_on = ['id']
file_out = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\merged_data_products\\merged_uc_r.25m_canopy_19_149_median-snow.csv'
data = merge_data_products(ddict, file_out, merge_data_in=merge_data_in, left_on=left_on, right_on=right_on, mode='median')

# 25cm products and median swe/dswe over upper forest
ddict = {
    'uf': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\synthetic_hemis\\hemi_grid_points\\mb_65_r.25m\\uf_plot_r.25m.tif',
    'swe_fcon_19_045': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_045\\19_045_las_proc\\OUTPUT_FILES\\SWE\\fcon\\interp_2x\\masked\\swe_fcon_19_045_r.05m_interp2x_masked.tif',
    'swe_fcon_19_050': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_050\\19_050_las_proc\\OUTPUT_FILES\\SWE\\fcon\\interp_2x\\masked\\swe_fcon_19_050_r.05m_interp2x_masked.tif',
    'swe_fcon_19_052': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_052\\19_052_las_proc\\OUTPUT_FILES\\SWE\\fcon\\interp_2x\\masked\\swe_fcon_19_052_r.05m_interp2x_masked.tif',
    'dswe_fnsd_19_045-19_050': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\mb_65\\dSWE\\fnsd\\interp_2x\\19_045-19_050\\masked\\dswe_fnsd_19_045-19_050_r.05m_interp2x_masked.tif',
    'dswe_fnsd_19_050-19_052': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\mb_65\\dSWE\\fnsd\\interp_2x\\19_050-19_052\\masked\\dswe_fnsd_19_050-19_052_r.05m_interp2x_masked.tif',
    ('er_p0_mean', 'er_p0_sd'): 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\ray_sampling\\batches\\lrs_mb_15_dem_.25m_61px_mp15.25\\outputs\\las_19_149_rs_mb_15_r.25_p0.0000_t3.1416.tif',
    'hemi_id': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\synthetic_hemis\\hemi_grid_points\\mb_65_1m\\1m_dem_point_ids.tif',
    'lrs_id': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\synthetic_hemis\\hemi_grid_points\\mb_65_r.25m\\dem_r.25_point_ids.tif'
}

merge_data_in = ['C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\synthetic_hemis\\batches\\mb_15_1m_pr.15_os10\\outputs\\LAI_parsed.dat',
                'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\ray_sampling\\batches\\lrs_uf_r.25_px181_snow_on\\outputs\\rshmetalog_footprint_products.csv']
left_on = ['hemi_id', 'lrs_id']
right_on = ['id', 'id']
file_out = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\merged_data_products\\merged_uf_r.25m_canopy_19_149_median-snow.csv'
data = merge_data_products(ddict, file_out, merge_data_in=merge_data_in, left_on=left_on, right_on=right_on, mode='median')


# 10cm products over mb_15
ddict = {
    'mb_15': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\synthetic_hemis\\hemi_grid_points\\mb_65_1m\\mb_15_plot_r.10m.tif',
    'plots': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\synthetic_hemis\\hemi_grid_points\\mb_65_r.25m\\site_plots_r.10m.tif',
    'dnt': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\DNT\\19_149_spike_free_dsm_can_r.10m_kho_distance_.10m.tif',
    'dce': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\DCE\\19_149_spike_free_chm_r.10m_dce.tif',
    'chm': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\CHM\\19_149_spike_free_chm_r.10m.tif',
    'lpmf15': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\LPM\\19_149_LPM-first_a15_r0.10m.tif',
    'lpml15': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\LPM\\19_149_LPM-last_a15_r0.10m.tif',
    'lpmc15': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\LPM\\19_149_LPM-canopy_a15_r0.10m.tif',
    'mCH_19_149': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\LAS\\19_149_las_proc_classified_merged_mCH.bil',
    'mCH_19_149_resampled': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\LAS\\19_149_las_proc_classified_merged_19_149_r0.25m_vox_resampled_mCH.bil',
    'mCH_045_050_052_resampled': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\ray_sampling\\sources\\045_050_052_combined_WGS84_utm11N_r0.25_vox_resampled_mCH.bil'
}
file_out = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\merged_data_products\\merged_mb_15_r.10m_canopy_19_149.csv'
merge_data_products(ddict, file_out)

# 10cm products and median swe/dswe over upper clearing
ddict = {
    'uc': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\synthetic_hemis\\hemi_grid_points\\mb_65_r.25m\\uc_plot_r.10m.tif',
    'swe_clin_19_045': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_045\\19_045_las_proc\\OUTPUT_FILES\\SWE\\clin\\interp_2x\\masked\\swe_clin_19_045_r.05m_interp2x_masked.tif',
    'swe_clin_19_050': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_050\\19_050_las_proc\\OUTPUT_FILES\\SWE\\clin\\interp_2x\\masked\\swe_clin_19_050_r.05m_interp2x_masked.tif',
    'swe_clin_19_052': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_052\\19_052_las_proc\\OUTPUT_FILES\\SWE\\clin\\interp_2x\\masked\\swe_clin_19_052_r.05m_interp2x_masked.tif',
    'dswe_cnsd_19_045-19_050': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\mb_65\\dSWE\\cnsd\\interp_2x\\19_045-19_050\\masked\\dswe_cnsd_19_045-19_050_r.05m_interp2x_masked.tif',
    'dswe_cnsd_19_050-19_052': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\mb_65\\dSWE\\cnsd\\interp_2x\\19_050-19_052\\masked\\dswe_cnsd_19_050-19_052_r.05m_interp2x_masked.tif',
    'dnt': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\DNT\\19_149_spike_free_dsm_can_r.10m_kho_distance_.10m.tif',
    'dce': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\DCE\\19_149_spike_free_chm_r.10m_dce.tif',
    'chm': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\CHM\\19_149_spike_free_chm_r.10m.tif',
    'lpmf15': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\LPM\\19_149_LPM-first_a15_r0.10m.tif',
    'lpml15': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\LPM\\19_149_LPM-last_a15_r0.10m.tif',
    'lpmc15': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\LPM\\19_149_LPM-canopy_a15_r0.10m.tif',
    'mCH_19_149': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\LAS\\19_149_las_proc_classified_merged_mCH.bil',
    'mCH_19_149_resampled': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\LAS\\19_149_las_proc_classified_merged_19_149_r0.25m_vox_resampled_mCH.bil',
    'mCH_045_050_052_resampled': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\ray_sampling\\sources\\045_050_052_combined_WGS84_utm11N_r0.25_vox_resampled_mCH.bil'
}
file_out = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\merged_data_products\\merged_uc_r.10m_canopy_19_149_median-snow.csv'
merge_data_products(ddict, file_out, mode='median')


# 10cm products and median swe/dswe over upper forest
ddict = {
    'uf': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\synthetic_hemis\\hemi_grid_points\\mb_65_r.25m\\uf_plot_r.10m.tif',
    'swe_fcon_19_045': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_045\\19_045_las_proc\\OUTPUT_FILES\\SWE\\fcon\\interp_2x\\masked\\swe_fcon_19_045_r.05m_interp2x_masked.tif',
    'swe_fcon_19_050': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_050\\19_050_las_proc\\OUTPUT_FILES\\SWE\\fcon\\interp_2x\\masked\\swe_fcon_19_050_r.05m_interp2x_masked.tif',
    'swe_fcon_19_052': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_052\\19_052_las_proc\\OUTPUT_FILES\\SWE\\fcon\\interp_2x\\masked\\swe_fcon_19_052_r.05m_interp2x_masked.tif',
    'dswe_fnsd_19_045-19_050': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\mb_65\\dSWE\\fnsd\\interp_2x\\19_045-19_050\\masked\\dswe_fnsd_19_045-19_050_r.05m_interp2x_masked.tif',
    'dswe_fnsd_19_050-19_052': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\mb_65\\dSWE\\fnsd\\interp_2x\\19_050-19_052\\masked\\dswe_fnsd_19_050-19_052_r.05m_interp2x_masked.tif',
    'dnt': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\DNT\\19_149_spike_free_dsm_can_r.10m_kho_distance_.10m.tif',
    'dce': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\DCE\\19_149_spike_free_chm_r.10m_dce.tif',
    'chm': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\CHM\\19_149_spike_free_chm_r.10m.tif',
    'lpmf15': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\LPM\\19_149_LPM-first_a15_r0.10m.tif',
    'lpml15': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\LPM\\19_149_LPM-last_a15_r0.10m.tif',
    'lpmc15': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\LPM\\19_149_LPM-canopy_a15_r0.10m.tif',
    'mCH_19_149': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\LAS\\19_149_las_proc_classified_merged_mCH.bil',
    'mCH_19_149_resampled': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\LAS\\19_149_las_proc_classified_merged_19_149_r0.25m_vox_resampled_mCH.bil',
    'mCH_045_050_052_resampled': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\ray_sampling\\sources\\045_050_052_combined_WGS84_utm11N_r0.25_vox_resampled_mCH.bil'
}
file_out = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\merged_data_products\\merged_uf_r.10m_canopy_19_149_median-snow.csv'
merge_data_products(ddict, file_out, mode='median')

# 10cm products and median swe/dswe over upper forest
ddict = {
    'uf': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\synthetic_hemis\\hemi_grid_points\\mb_65_r.25m\\uf_plot_r.10m.tif',
    '19_045_pdens': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_045\\19_045_las_proc\\OUTPUT_FILES\\RAS\\19_045_ground_point_density_r.10m.bil',
    '19_050_pdens': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_050\\19_050_las_proc\\OUTPUT_FILES\\RAS\\19_050_ground_point_density_r.10m.bil',
    '19_052_pdens': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_052\\19_052_las_proc\\OUTPUT_FILES\\RAS\\19_052_ground_point_density_r.10m.bil',
    '19_149_pdens': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\RAS\\19_149_ground_point_density_r.10m.bil',
    '19_045_hs': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_045\\19_045_las_proc\\OUTPUT_FILES\\HS\\interp_2x\\clean\\19_045_hs_r.05m_interp2x_clean.tif',
    '19_050_hs': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_050\\19_050_las_proc\\OUTPUT_FILES\\HS\\interp_2x\\clean\\19_050_hs_r.05m_interp2x_clean.tif',
    '19_052_hs': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_052\\19_052_las_proc\\OUTPUT_FILES\\HS\\interp_2x\\clean\\19_052_hs_r.05m_interp2x_clean.tif',
    'lpml15': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\LPM\\19_149_LPM-last_a15_r0.10m.tif',
}
file_out = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\merged_data_products\\merged_uf_r.10m_point_density_median_snow_depth.csv'
merge_data_products(ddict, file_out, mode='median')

# 5cm snow products and nearest lpml15 over upper clearing
ddict = {
    'uc': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\synthetic_hemis\\hemi_grid_points\\mb_65_r.25m\\uc_plot_r.05m.tif',
    'swe_clin_19_045': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_045\\19_045_las_proc\\OUTPUT_FILES\\SWE\\clin\\interp_2x\\masked\\swe_clin_19_045_r.05m_interp2x_masked.tif',
    'swe_clin_19_050': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_050\\19_050_las_proc\\OUTPUT_FILES\\SWE\\clin\\interp_2x\\masked\\swe_clin_19_050_r.05m_interp2x_masked.tif',
    'swe_clin_19_052': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_052\\19_052_las_proc\\OUTPUT_FILES\\SWE\\clin\\interp_2x\\masked\\swe_clin_19_052_r.05m_interp2x_masked.tif',
    'dswe_cnsd_19_045-19_050': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\mb_65\\dSWE\\cnsd\\interp_2x\\19_045-19_050\\masked\\dswe_cnsd_19_045-19_050_r.05m_interp2x_masked.tif',
    'dswe_cnsd_19_050-19_052': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\mb_65\\dSWE\\cnsd\\interp_2x\\19_050-19_052\\masked\\dswe_cnsd_19_050-19_052_r.05m_interp2x_masked.tif',
    'lpml15': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\LPM\\19_149_LPM-last_a15_r0.10m.tif',
}
file_out = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\merged_data_products\\merged_uc_.05m_snow_nearest_canopy_19_149.csv'
merge_data_products(ddict, file_out, mode='nearest')

# 5cm snow products and nearest lpml15 over upper forest
ddict = {
    'uf': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\synthetic_hemis\\hemi_grid_points\\mb_65_r.25m\\uf_plot_r.05m.tif',
    'swe_fcon_19_045': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_045\\19_045_las_proc\\OUTPUT_FILES\\SWE\\fcon\\interp_2x\\masked\\swe_fcon_19_045_r.05m_interp2x_masked.tif',
    'swe_fcon_19_050': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_050\\19_050_las_proc\\OUTPUT_FILES\\SWE\\fcon\\interp_2x\\masked\\swe_fcon_19_050_r.05m_interp2x_masked.tif',
    'swe_fcon_19_052': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_052\\19_052_las_proc\\OUTPUT_FILES\\SWE\\fcon\\interp_2x\\masked\\swe_fcon_19_052_r.05m_interp2x_masked.tif',
    'dswe_fnsd_19_045-19_050': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\mb_65\\dSWE\\fnsd\\interp_2x\\19_045-19_050\\masked\\dswe_fnsd_19_045-19_050_r.05m_interp2x_masked.tif',
    'dswe_fnsd_19_050-19_052': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\mb_65\\dSWE\\fnsd\\interp_2x\\19_050-19_052\\masked\\dswe_fnsd_19_050-19_052_r.05m_interp2x_masked.tif',
    'lpml15': 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\LPM\\19_149_LPM-last_a15_r0.10m.tif',
}
file_out = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\merged_data_products\\merged_uf_.05m_snow_nearest_canopy_19_149.csv'
merge_data_products(ddict, file_out, mode='nearest')
