def main():
    import las_ray_sampling as lrs
    import numpy as np
    import pandas as pd
    import os

    # call voxel config
    # import vox_045_050_052_config as vc
    import vox_19_149_config as vc
    vox = vc.vox


    batch_dir = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\ray_sampling\\batches\\lrs_photo_initial_test\\'
    # pts_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\synthetic_hemis\\hemi_grid_points\\mb_65_r.25m\\dem_r.25_points_uf.csv"


    # batch_dir = 'C:\\Users\\jas600\\workzone\\data\\ray_sampling\\batches\\lrs_uf_r.25_px181_snow_off_dem_offset.25\\'
    # pts_in = 'C:\\Users\\jas600\\workzone\\data\\hemi_grid_points\\mb_65_r.25m_snow_off_offset.25\\dem_r.25_points_uf.csv'

    # load points
    # pts = pd.read_csv(pts_in)
    pts = np.array([[628151.1, 5646577.9, 1830.7406]])

    rspmeta = lrs.RaySamplePhotoMetaObj()

    # rshmeta.lookup_db = 'count'
    rspmeta.lookup_db = 'posterior'
    rspmeta.config_id = vc.config_id

    rspmeta.agg_sample_length = vox.sample_length
    rspmeta.agg_method = 'single_ray_agg'

    # ray geometry


    photo_m_above_ground = 0  # meters
    rspmeta.max_distance = 50  # meters
    rspmeta.min_distance = 0  # vox.step[0] * np.sqrt(3)  # meters
    rspmeta.phi_range = np.array([75, 90]) * np.pi / 180
    rspmeta.phi_count = np.array([100])
    rspmeta.theta_count = np.array([160])
    # maintain square aspect ratio
    # this is not totally accurate, more error further from horizon... can adress with spherical rotation prior to mercator projeciton, then rerotate.
    theta_delta = np.diff(rspmeta.phi_range) * rspmeta.theta_count / rspmeta.phi_count
    theta_mid = 0
    rspmeta.theta_range = np.array([theta_mid - theta_delta / 2, theta_mid + theta_delta / 2]).swapaxes(0, 1)



    # create batch dir (with error handling)
    # if batch file dir exists
    if os.path.exists(batch_dir):
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
    rspmeta.file_dir = batch_dir + "outputs\\"
    if not os.path.exists(rspmeta.file_dir):
        os.makedirs(rspmeta.file_dir)


    # rspmeta.id = pts.id
    rspmeta.id = np.array([0])
    rspmeta.origin = pts
    rspmeta.origin[:, 2] = rspmeta.origin[:, 2] + photo_m_above_ground

    rspmeta.file_name = ["las_photo_id_" + str(xx) + ".tif" for xx in rspmeta.id]

    rspm = lrs.rs_photogen(rspmeta, vox)

    ###
    #
    # import matplotlib
    # matplotlib.use('TkAgg')
    # import matplotlib.pyplot as plt
    # import tifffile as tif
    # ii = 0
    # peace = tif.imread(rshm.file_dir[ii] + rshm.file_name[ii])
    # plt.imshow(peace[:, :, 2], interpolation='nearest')

if __name__ == "__main__":
    main()
