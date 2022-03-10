def main():
    """
    Configuration file for generating synthetic hemispheres corresponding to georeferenced hemispherical photography,
    for subsequent validation/determination of contact number scaling coefficient
        batch_dir: directory for all batch outputs
        img_lookup_in: file of processed hemispherical photographs
        pts_in: coordinates and elevations at which to calculate hemispheres

    :return:
    """

    from libraries import las_ray_sampling as lrs
    import numpy as np
    import pandas as pd
    import os

    # call voxel config
    # import voxel_ray_sampling_of_lidar.vox_045_050_052_config as vc  # snow on canopy
    import voxel_ray_sampling_of_lidar.vox_19_149_config as vc  # snow off canopy
    vox = vc.vox

    batch_dir = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\ray_sampling\\batches\\timing\\lrs_hemi_optimization_r.25_px181_snow_off\\'

    # batch_dir = 'C:\\Users\\jas600\\workzone\\data\\ray_sampling\\batches\\lrs_hemi_opt_test\\'


    img_lookup_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\hemispheres\\hemi_lookup_cleaned.csv"
    # img_lookup_in = 'C:\\Users\\jas600\\workzone\\data\\las\\hemi_lookup_cleaned.csv'

    # filter imges based on quality control codes/day, etc.
    max_quality = 4
    # las_day = ["19_045", "19_050", "19_052"]
    las_day = ["19_149"]
    # import hemi_lookup
    img_lookup = pd.read_csv(img_lookup_in)
    # filter lookup by quality
    img_lookup = img_lookup[img_lookup.quality_code <= max_quality]
    # filter lookup by las_day
    img_lookup = img_lookup[np.in1d(img_lookup.folder, las_day)]

    [file.replace('.JPG', '') for file in img_lookup.filename]


    pts = pd.DataFrame({'id': img_lookup.filename,
                        'x_utm11n': img_lookup.xcoordUTM1,
                        'y_utm11n': img_lookup.ycoordUTM1,
                        'z_m': img_lookup.elevation})


    rshmeta = lrs.RaySampleGridMetaObj()

    # rshmeta.lookup_db = 'count'
    rshmeta.lookup_db = 'posterior'
    rshmeta.config_id = vc.config_id

    rshmeta.agg_method = 'single_ray_agg'
    rshmeta.agg_sample_length = vox.agg_sample_length


    # rshmeta.agg_method = 'vox_agg'  # vox_shell_agg?
    # rshmeta.agg_sample_length = vox.step[0] ** 2/vox.sample_length
    # rshmeta.min_distance = 0

    # rshmeta.agg_method = 'multi_ray_agg'


    # ray geometry
    # phi_step = (np.pi / 2) / (180 * 2)
    rshmeta.img_size = 181  # square, in pixels/ray samples
    # rshmeta.max_phi_rad = phi_step * rshmeta.img_size
    rshmeta.max_phi_rad = np.pi/2
    hemi_m_above_ground = img_lookup.height_m  # meters
    rshmeta.max_distance = 50  # meters
    # rshmeta.min_distance = vox.step[0] * np.sqrt(3)  # meters
    rshmeta.min_distance = 0  # meters

    # create batch dir
    # if batch file dir exists
    if os.path.exists(batch_dir):
        input_needed = True
        while input_needed:
            batch_exist_action = input("Batch file directory already exists. Would you like to: (P)roceed, (E)rase and proceed, or (A)bort?")
            if batch_exist_action.upper() == "P":
                input_needed = False
            elif batch_exist_action.upper() == "E":
                file_count = sum(len(files) for _, _, files in os.walk(batch_dir))  # dir is your directory path as string
                remove_confirmation = input("\tConfirm: Overwrite batch folder with " + str(file_count) + " contained files? (Y/N)")
                if remove_confirmation.upper() == "Y":
                    # delete folder and contents
                    import shutil
                    shutil.rmtree(batch_dir)
                    # recreate dir
                    os.makedirs(batch_dir)
                    input_needed = False
                else:
                    # return to while loop
                    pass
            elif batch_exist_action.upper() == "A":
                raise Exception("Execution aborted by user input.")
            else:
                print("Invalid user input.")
    else:
        # create dir
        os.makedirs(batch_dir)

    # create output file dir
    rshmeta.file_dir = batch_dir + "outputs\\"
    if not os.path.exists(rshmeta.file_dir):
        os.makedirs(rshmeta.file_dir)

    rshmeta.id = pts.id
    rshmeta.origin = np.array([pts.x_utm11n,
                               pts.y_utm11n,
                               pts.z_m + hemi_m_above_ground]).swapaxes(0, 1)

    rshmeta.file_name = ["las_19_149_id_" + str(id) + ".tif" for id in pts.id]

    rshm = lrs.rs_hemigen(rshmeta, vox, initial_index=0)

if __name__ == "__main__":
    main()
#
# # preliminary visualization
# import numpy as np
# import pandas as pd
# import matplotlib
# matplotlib.use('Qt5Agg')
# import matplotlib.pyplot as plt
# import tifffile as tif
#
#
# def load_lrs_img_cn(batch_dir, coef, ii):
#     rshmeta = pd.read_csv(batch_dir + "outputs\\rshmetalog.csv")
#     img = tif.imread(batch_dir + "outputs\\" + rshmeta.file_name[ii])
#     return img[:, :, 0] * coef
#
# for ii in range(0, 58):
# ii = 14
#
#     # snow_off_dir = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\ray_sampling\\batches\\lrs_uf_r.25_px181_snow_off_dem_offset.25\\"
#     snow_off_dir = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\ray_sampling\\batches\\lrs_hemi_optimization_r.25_px1000_snow_off\\"
#     snow_off_coef = 0.38686933  # optimized for tx dropping 5th ring
#     # snow_off_coef = 0.19216  # optimized for cn dropping 5th
#     # snow_off_coef = 0.191206
#     # snow_off_coef = 0.155334
#     # snow_off_coef = 0.1841582  # tx wls
#     # snow_off_coef = 0.220319  # cn wls
#     # snow_off_coef = 0.1857892  # tx wmae
#     # snow_off_coef = 0.2137436  # cn wmae
#
#     cn_off = load_lrs_img_cn(snow_off_dir, snow_off_coef, ii)
#     tx_off = np.exp(-cn_off)
#     #
#     # # snow_on_dir = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\ray_sampling\\batches\\lrs_uf_r.25_px181_snow_on_dem_offset.25\\"
#     # snow_on_dir = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\ray_sampling\\batches\\lrs_hemi_optimization_r.25_px1000_snow_on\\"
#     # # snow_on_dir = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\ray_sampling\\batches\\lrs_hemi_optimization_r.25_px1000_snow_on_at_snow_off\\"
#     # snow_on_coef = 0.37181197  # python tx dropping 5th
#     # # snow_on_coef = 0.136461  # optimized for cn dropping 5th
#     # # snow_on_coef = 0.132154
#     # # snow_on_coef = 0.137942
#     # # snow_on_coef = 0.169215  # tx wls
#     # # snow_on_coef = 0.141832  # cn wls
#     # # snow_on_coef = 0.1736879  # tx wmae
#     # # snow_on_coef = 0.1487048  # cn wmae
#     # cn_on = load_lrs_img_cn(snow_on_dir, snow_on_coef, ii)
#     # tx_on = np.exp(-cn_on)
#     #
#     #
#     # ##
#     plot_out_dir = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\graphics\\thesis_graphics\\hemispheres\\"
#
#
#     fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
#     # fig, ax = plt.subplots(figsize=(1.81, 1.81), dpi=100)
#     # img = ax.imshow(tx_off, interpolation='nearest', cmap='Greys_r', clim=[0, 1])
#     fim = plt.figimage(tx_off, cmap='Greys_r', clim=[0, 1])
#     ax.set_axis_off()
#     # fig.savefig(plot_out_dir + 'lrs_snow_on_tx_id' + str(ii) + '.png', bbox_inches='tight', pad_inches=0)
#     fig.savefig(plot_out_dir + 'lrs_snow_off_tx_id' + str(ii) + '.png')
# #
#     # fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
#     # # fig, ax = plt.subplots(figsize=(1.81, 1.81), dpi=100)
#     # # img = ax.imshow(tx_on, interpolation='nearest', cmap='Greys_r', clim=[0, 1])
#     # fim = plt.figimage(tx_on, cmap='Greys_r', clim=[0, 1])
#     # ax.set_axis_off()
#     # fig.savefig(plot_out_dir + 'lrs_snow_on_tx_id' + str(ii) + '.png', bbox_inches='tight', pad_inches=0)
#     # fig.savefig(plot_out_dir + 'lrs_snow_on_tx_id' + str(ii) + '.png')
# #
# #
