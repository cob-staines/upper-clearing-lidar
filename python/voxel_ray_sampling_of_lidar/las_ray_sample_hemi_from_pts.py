def main():
    """
    Configuration file for generating synthetic hemispheres of expected lidar returns by voxel ray samping of lidar
    (eg. to estimate light transmittance across the hemisphere at a given point, Staines thesis figure 3.3).
        batch_dir: directory for all batch outputs
        pts_in: coordinates and elevations at which to calculate hemispheres

    :return:
    """

    from libraries import las_ray_sampling as lrs
    import numpy as np
    import pandas as pd
    import os

    # call voxel config
    # import vox_045_050_052_config as vc  # snow on canopy
    import vox_19_149_config as vc  # snow off canopy
    vox = vc.vox


    # batch_dir = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\ray_sampling\\batches\\lrs_mb_15_r.25_px181_test\\'
    # pts_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\synthetic_hemis\\hemi_grid_points\\mb_65_r.25m\\dem_r.25_points_uf.csv"


    batch_dir = 'C:\\Users\\jas600\\workzone\\data\\ray_sampling\\batches\\lrs_uf_r.25_px181_snow_off_dem_offset.25\\'
    pts_in = 'C:\\Users\\jas600\\workzone\\data\\hemi_grid_points\\mb_65_r.25m_snow_off_offset.25\\dem_r.25_points_uf.csv'

    # load points
    pts = pd.read_csv(pts_in)

    rshmeta = lrs.RaySampleGridMetaObj()

    # rshmeta.lookup_db = 'count'
    rshmeta.lookup_db = 'posterior'
    rshmeta.config_id = vc.config_id

    rshmeta.agg_sample_length = vox.sample_length
    rshmeta.agg_method = 'single_ray_agg'

    # ray geometry
    # phi_step = (np.pi / 2) / (180 * 2)
    rshmeta.img_size = 181  # angular resolution (square) ~ samples / pi
    # rshmeta.max_phi_rad = phi_step * rshmeta.img_size
    rshmeta.max_phi_rad = np.pi/2  # mazimum zenith angle to sample
    hemi_m_above_ground = 0  # height above ground of synthetic hemisphere in meters
    rshmeta.max_distance = 50  # maximum distance to sample ray in meters
    rshmeta.min_distance = 0  # minimum distance to sample ray in meters

    # create batch dir (with error handling)
    if os.path.exists(batch_dir):
        # if batch file dir exists
        input_needed = True
        while input_needed:
            batch_exist_action = input("Batch file directory already exists. Would you like to: (P)roceed, (E)rase and proceed, or (A)bort? ")
            if batch_exist_action.upper() == "P":
                input_needed = False
            elif batch_exist_action.upper() == "E":
                file_count = sum(len(files) for _, _, files in os.walk(batch_dir))  # dir is your directory path as string
                remove_confirmation = input("Remove batch folder with " + str(file_count) + " contained files (Y/N)? ")
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

    rshm = lrs.rs_hemigen(rshmeta, vox, tile_count_1d=5, n_cores=3)


if __name__ == "__main__":
    main()

# # preliminary visualization
#
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# import tifffile as tif
# ii = 0
# peace = tif.imread(rshm.file_dir[ii] + rshm.file_name[ii])
# plt.imshow(peace[:, :, 2], interpolation='nearest')