def main():
    """
    Configuration file for generating raster of expected lidar returns by voxel ray samping of lidar over a set of grid
    points and for a specified angle (or sets of angles) of the hemisphere (eg. to estimate contact number across the
    site at a given angle, Staines thesis figure 4.5). Use this when only specific angles are needed. If angles across
    the hemisphere are needed, use las_ray_sample_hemi_from_pts.py instead.
        batch_dir: directory for all batch outputs
        dem_in: raster template and elevations to be used for grid
        mask_in: binary raster (from grid_points.py) of points to be used

    :return:
    """

    from libraries import las_ray_sampling as lrs
    import numpy as np
    import os

    # call voxel config
    import vox_19_149_config as vc  # snow off canopy
    # import vox_045_050_052_config as vc  # snow on canopy
    vox = vc.vox


    batch_dir = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\ray_sampling\\batches\\lrs_mb_15_dem_.25m_snow_off_single_runs\\'
    dem_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\DEM\\interpolated\\19_149_dem_interpolated_r.25m.tif"
    mask_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\site_library\\hemi_grid_points\\mb_65_r.25m_snow_off_offset.25\\mb_15_plot_r.25m.tif"

    # batch_dir = 'C:\\Users\\jas600\\workzone\\data\\ray_sampling\\\\batches\\lrs_mb_15_dem_.25m_61px_mp15.25\\'
    # dem_in = "C:\\Users\\jas600\\workzone\\data\\ray_sampling\\19_149_dem_interpolated_r.25m.tif"
    # mask_in = "C:\\Users\\jas600\\workzone\\data\\ray_sampling\\mb_15_plot_r.25m.tif"


    rsgmeta = lrs.RaySampleGridMetaObj()

     # rshmeta.lookup_db = 'count'  # troubleshooting only
    rsgmeta.lookup_db = 'posterior'  # explicitly sample from bayesian posterior
    rsgmeta.config_id = vc.config_id

    rsgmeta.agg_sample_length = vox.sample_length
    rsgmeta.agg_method = 'single_ray_agg'  # always use this (other options were never fully developed)

    # define input files
    rsgmeta.src_ras_file = dem_in
    rsgmeta.mask_file = mask_in

    # create batch dir
    if os.path.exists(batch_dir):
        # if batch file dir exists
        input_needed = True
        while input_needed:
            batch_exist_action = input("Batch file directory already exists. Would you like to: (P)roceed, (E)rase and proceed, or (A)bort?")
            if batch_exist_action.upper() == "P":
                input_needed = False
            elif batch_exist_action.upper() == "E":
                file_count = sum(len(files) for _, _, files in os.walk(batch_dir))  # dir is your directory path as string
                remove_confirmation = input("Remove batch folder with " + str(file_count) + " contained files? (Y/N)")
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
    rsgmeta.file_dir = batch_dir + "outputs\\"
    if not os.path.exists(rsgmeta.file_dir):
        os.makedirs(rsgmeta.file_dir)

    # ray geometry
    rsgmeta.max_distance = 50  # meters
    rsgmeta.min_distance = vox.step[0] * np.sqrt(3)  # meters

    # calculate hemisphere of phi and theta values
    rsgmeta.phi = np.pi / 8  # zenith angle of rays (radians)
    rsgmeta.theta = 3 * np.pi / 4  # azimuth angle of rays (clockwise from north, from above looking down, in radians)
    rsgmeta.id = 0
    rsgmeta.file_name = ["las_19_149_rs_mb_15_r.25_p{:.4f}_t{:.4f}.tif".format(rsgmeta.phi, rsgmeta.theta)]


    # phi_step = (np.pi / 2) / (180 * 2)
    # rsgmeta.set_phi_size = 1  # square, in pixels/ray samples
    # rsgmeta.set_max_phi_rad = phi_step * rsgmeta.set_phi_size
    # rsgmeta.set_max_phi_rad = np.pi/2

    # export phi_theta_lookup of vectors in grid
    # vector_set = lrs.hemi_vectors(rsgmeta.set_phi_size, rsgmeta.set_max_phi_rad).sort_values('phi').reset_index(drop=True)
    # vector_set.to_csv(rsgmeta.file_dir + "phi_theta_lookup.csv", index=False)
    #
    # rsgmeta.phi = vector_set.phi.values
    # rsgmeta.theta = vector_set.theta.values
    # rsgmeta.id = vector_set.index.values
    # rsgmeta.file_name = ["las_19_149_rs_mb_15_r.25_p{:.4f}_t{:.4f}.tif".format(rsgmeta.phi[ii], rsgmeta.theta[ii]) for ii in rsgmeta.id]

    rsgm = lrs.rs_gridgen(rsgmeta, vox, initial_index=0)


if __name__ == "__main__":
    main()


##


# # preliminary visualization
#
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# import tifffile as tif
# ii = 0
# peace = tif.imread(rshm.file_dir[ii] + rshm.file_name[ii])
# plt.imshow(peace[:, :, 2], interpolation='nearest')