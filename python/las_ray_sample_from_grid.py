def main():
    import las_ray_sampling as lrs
    import numpy as np
    import pandas as pd
    import os

    # build voxel space
    import vox_19_149_config as vc
    vox = vc.vox
    vox = lrs.las_to_vox(vox, vc.z_slices, run_las_traj=False, fail_overflow=False, calc_prior=True)


    # # LOAD VOX
    print('Loading vox... ', end='')
    vox = lrs.load_vox_meta(vox.vox_hdf5, load_data=False)
    print('done')


    batch_dir = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\ray_sampling\\batches\\lrs_mb_15_dem_.25m_single_runs_test\\'
    dem_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\DEM\\interpolated\\19_149_dem_interpolated_r.25m.tif"
    mask_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\synthetic_hemis\\hemi_grid_points\\mb_65_1m\\mb_15_plot_r.25m.tif"

    # batch_dir = 'C:\\Users\\jas600\\workzone\\data\\ray_sampling\\\\batches\\lrs_mb_15_dem_.25m_61px_mp15.25\\'
    # dem_in = "C:\\Users\\jas600\\workzone\\data\\ray_sampling\\19_149_dem_interpolated_r.25m.tif"
    # mask_in = "C:\\Users\\jas600\\workzone\\data\\ray_sampling\\mb_15_plot_r.25m.tif"


    rsgmeta = lrs.RaySampleGridMetaObj()

    rsgmeta.agg_sample_length = vox.sample_length
    rsgmeta.agg_method = 'beta_lookup'

    print('Calculating prior... ', end='')
    if rsgmeta.agg_method == 'nb_lookup':
        mean_path_length = 2 * np.pi / (6 + np.pi) * voxel_length  # mean path length through a voxel cube across angles (m)
        prior_weight = 5  # in units of scans (1 <=> equivalent weight to 1 expected voxel scan)
        prior_b = mean_path_length * prior_weight
        prior_a = prior_b * 0.01
        rsgmeta.prior = [prior_a, prior_b]
        rsgmeta.ray_iterations = 100  # model runs for each ray, from which median and std of returns is calculated
    elif rsgmeta.agg_method == 'linear':
        samps = (vox.sample_data > 0)
        trans = vox.return_data[samps] // (vox.sample_data[samps] * vox.sample_length)
        rsgmeta.prior = np.var(trans)
    elif rsgmeta.agg_method == 'beta':
        val = (vox.sample_data > 0)  # roughly 50% at .25m
        rate = vox.return_data[val] / vox.sample_data[val]
        mu = np.mean(rate)
        sig2 = np.var(rate)

        alpha = ((1 - mu)/sig2 - 1/mu) * (mu ** 2)
        beta = alpha * (1/mu - 1)
        rsgmeta.prior = [alpha, beta]
    elif rsgmeta.agg_method == 'beta_lookup':
        # lrs.beta_lookup_prior_calc(vox, rshmeta.ray_sample_length)
        pass
    else:
        raise Exception('Aggregation method ' + rsgmeta.agg_method + ' unknown.')
    print('done')

    # ray geometry
    phi_step = (np.pi / 2) / (180 * 2)
    rsgmeta.set_phi_size = 61  # square, in pixels/ray samples
    rsgmeta.set_max_phi_rad = phi_step * rsgmeta.set_phi_size
    # rsgmeta.set_max_phi_rad = np.pi/2
    # ray m above ground?
    rsgmeta.max_distance = 50  # meters
    rsgmeta.min_distance = voxel_length * np.sqrt(3)  # meters

    # define input files
    rsgmeta.src_ras_file = dem_in
    rsgmeta.mask_file = mask_in


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


    # calculate hemisphere of phi and theta values
    rsgmeta.phi = np.pi / 8
    rsgmeta.theta = 3 * np.pi / 4
    rsgmeta.id = 0
    rsgmeta.file_name = ["las_19_149_rs_mb_15_r.25_p{:.4f}_t{:.4f}.tif".format(rsgmeta.phi, rsgmeta.theta)]

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


# min_dist = rsgmeta.min_distance
# max_dist = rsgmeta.max_distance



# ###
# #
# # import matplotlib
# # matplotlib.use('TkAgg')
# # import matplotlib.pyplot as plt
# # import tifffile as tif
# # ii = 0
# # peace = tif.imread(rshm.file_dir[ii] + rshm.file_name[ii])
# # plt.imshow(peace[:, :, 2], interpolation='nearest')