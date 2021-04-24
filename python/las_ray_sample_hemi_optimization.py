def main():
    import las_ray_sampling as lrs
    import numpy as np
    import pandas as pd
    import os

    # call voxel config
    import vox_19_149_config as vc
    vox = vc.vox

    # batch_dir = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\ray_sampling\\batches\\lrs_hemi_optimization_r.25_px181_beta_single_ray_agg_19_149\\'
    batch_dir = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\ray_sampling\\batches\\lrs_hemi_optimization_r.25_px1000_snow_off\\'

    # batch_dir = 'C:\\Users\\jas600\\workzone\\data\\ray_sampling\\batches\\lrs_hemi_opt_test\\'


    img_lookup_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\hemispheres\\hemi_lookup_cleaned.csv"
    # img_lookup_in = 'C:\\Users\\jas600\\workzone\\data\\las\\hemi_lookup_cleaned.csv'
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

    rshmeta.agg_method = 'single_ray_agg'
    rshmeta.agg_sample_length = vox.agg_sample_length


    # rshmeta.agg_method = 'vox_agg'  # vox_shell_agg?
    # rshmeta.agg_sample_length = vox.step[0] ** 2/vox.sample_length
    # rshmeta.min_distance = 0

    # rshmeta.agg_method = 'multi_ray_agg'


    # ray geometry
    # phi_step = (np.pi / 2) / (180 * 2)
    rshmeta.img_size = 1000  # square, in pixels/ray samples
    # rshmeta.max_phi_rad = phi_step * rshmeta.img_size
    rshmeta.max_phi_rad = np.pi/2
    hemi_m_above_ground = img_lookup.height_m  # meters
    rshmeta.max_distance = 50  # meters
    rshmeta.min_distance = vox.step[0] * np.sqrt(3)  # meters

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

    rshm = lrs.rs_hemigen(rshmeta, vox)




if __name__ == "__main__":
    main()
#
# #
# import numpy as np
# import pandas as pd
# import matplotlib
# matplotlib.use('Qt5Agg')
# import matplotlib.pyplot as plt
# import tifffile as tif
#
#
# # load rshmetalog
# batch_dir = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\ray_sampling\\batches\\lrs_hemi_optimization_r.25_px1000\\"
# # batch_dir = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\ray_sampling\\batches\\lrs_hemi_optimization_r.25_px181_beta_single_ray_agg_045_050_052\\'
# # batch_dir = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\ray_sampling\\batches\\lrs_uf_r.25_px181_snow_on\\"
# # batch_dir = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\ray_sampling\\batches\\lrs_uf_r.25_px181_snow_on\\"
#
# rshmeta = pd.read_csv(batch_dir + "outputs\\rshmetalog.csv")
#
# ii = 0
#
# img = tif.imread(batch_dir + "outputs\\" + rshmeta.file_name[ii])
# # rt = img[:, :, 0]
# cn = img[:, :, 0] * 0.198508  # snow_off
# # cn = img[:, :, 0] * 0.132154  # snow_on
# tx = np.exp(-cn)
#
#
# ##
#
#
# fig, ax = plt.subplots(figsize=(12, 12))
# img = ax.imshow(tx, interpolation='nearest', cmap='Greys_r')
# ax.set_axis_off()
#
# # fig.savefig(batch_dir + 'light_transmission_plot_' + rshmeta.file_name[ii] + '.png')
