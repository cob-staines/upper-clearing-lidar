import subprocess
import rastools
import laspy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

lastools_dir = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\code_lib\\lastools\\LAStools\\bin\\"
plot_out_dir = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\graphics\\thesis_graphics\\frequency distributions\\"

def mCH(las_file):
    norm_file = las_file.replace(".las", "_normalized.las")
    noise_file = las_file.replace(".las", "_noise.las")
    mch_file = las_file.replace(".las", "_mCH.bil")

    epsg = '32611'
    res = str(0.10)
    gp_file = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\LAS\\19_149_las_proc_ground_thinned_merged.las"

    noise_class = str(7)
    noise_isolation = str(2)
    noise_step = str(0.25)

    z_min = str(1)
    z_max = str(30)

    # normalize heights
    lasheight_cmd = [lastools_dir + "lasheight.exe",
                    '-i', las_file,
                    '-epsg', epsg,
                    '-ground_points', gp_file,
                    '-all_ground_points',
                    '-replace_z',
                    '-o', norm_file]

    pcs = subprocess.Popen(lasheight_cmd)  # start process
    pcs.wait()  # wait for it to finish

    # normalize heights
    lasnoise_cmd = [lastools_dir + "lasnoise.exe",
                    '-i', norm_file,
                    '-epsg', epsg,
                    '-isolated', noise_isolation,
                    '-step', noise_step,
                    '-classify_as', noise_class,
                    '-o', noise_file]

    pcs = subprocess.Popen(lasnoise_cmd)  # start process
    pcs.wait()  # wait for it to finish

    # calculate mean canopy height
    lasgrid_cmd = [lastools_dir + "lasgrid.exe",
                '-i', noise_file,
                '-epsg', epsg,
                '-keep_z', z_min, z_max,
                '-drop_class', noise_class,
                '-step', res,
                '-elevation',
                '-mean',
                '-o', mch_file]

    pcs = subprocess.Popen(lasgrid_cmd)  # start process
    pcs.wait()  # wait for it to finish

    ras = rastools.raster_load(mch_file)
    ras.data[ras.data == ras.no_data] = 0
    rastools.raster_save(ras, mch_file, file_format="EHdr")


def las_clip(las_file, shp_file):
    # clip to shp file
    epsg = '32611'
    clip_file = las_file.replace(".las", "_clipped.las")
    lasclip_cmd = [lastools_dir + "lasclip.exe",
                   '-i', las_file,
                   '-epsg', epsg,
                   '-poly', shp_file,
                   '-o', clip_file]

    pcs = subprocess.Popen(lasclip_cmd)  # start process
    pcs.wait()  # wait for it to finish


# canopy height distribution (LAS)
def comparison_histogram(las_list, bins="auto"):

    mean_z = []

    for ii in range(len(las_list)):

        print('Loading LAS file... ', end='')
        # load las_in
        inFile = laspy.file.File(las_list[ii], mode="r")
        # load data
        p0 = pd.DataFrame({"x": inFile.x,
                           "y": inFile.y,
                           "z": inFile.z,
                           "classification": inFile.classification})
        # close las_in
        inFile.close()
        print('done')

        # drop noise & points below 1m
        valid = (p0.z >= 1) & (p0.classification != 7)

        p0 = p0.assign(file=ii)
        p0 = p0.loc[valid, ["z", "file"]]

        # print(str(np.mean(p0.z)))
        if ii == 0:
            composite = p0
        else:
            composite = pd.concat([composite, p0])

        mean_z.append(np.mean(p0.z))

    # plot histogram of z
    plot = sns.histplot(composite, x="z", bins=bins, stat="density", hue="file", common_norm=False, element="step")
    return plot, mean_z


las_in = ["C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\ray_sampling\\sources\\045_050_052_combined_WGS84_utm11N_r0.25_vox_resampled.las",
          'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\LAS\\19_149_las_proc_classified_merged_19_149_r0.25m_vox_resampled.las',
          r"C:\Users\Cob\index\educational\usask\research\masters\data\lidar\19_149\19_149_las_proc\OUTPUT_FILES\LAS\19_149_las_proc_classified_merged.las",
          r"C:\Users\Cob\index\educational\usask\research\masters\data\lidar\19_149\19_149_las_proc\OUTPUT_FILES\LAS\19_149_las_proc_classified_merged_poisson_0.05.las",
          r"C:\Users\Cob\index\educational\usask\research\masters\data\lidar\19_149\19_149_las_proc\OUTPUT_FILES\LAS\19_149_las_proc_classified_merged_poisson_0.15.las"]

ff = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\LAS\\19_149_las_proc_classified_merged_19_149_r0.25m_vox_resampled.las'

for ff in las_in:
    mCH(ff)

shp_file = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\site_library\\upper_forest_poly_UTM11N.shp'
for ff in las_in:
    ff_noise = ff.replace(".las", "_noise.las")
    las_clip(ff_noise, shp_file)


uf_ch_list = [r"C:\Users\Cob\index\educational\usask\research\masters\data\lidar\19_149\19_149_las_proc\OUTPUT_FILES\LAS\19_149_las_proc_classified_merged_noise_clipped.las",
              # r"C:\Users\Cob\index\educational\usask\research\masters\data\lidar\19_149\19_149_las_proc\OUTPUT_FILES\LAS\19_149_las_proc_classified_merged_poisson_0.05_noise_clipped.las",
              # r"C:\Users\Cob\index\educational\usask\research\masters\data\lidar\19_149\19_149_las_proc\OUTPUT_FILES\LAS\19_149_las_proc_classified_merged_poisson_0.15_noise_clipped.las",
              'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\LAS\\19_149_las_proc_classified_merged_19_149_r0.25m_vox_resampled_noise_clipped.las',
              # "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\ray_sampling\\sources\\045_050_052_combined_WGS84_utm11N_r0.25_vox_resampled_noise_clipped.las"
]

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Canopy lidar return height frequency for snow-free canopy\n Upper Forest plot, 29 May 2019')
ax1.set_xlabel("Height above ground [m]")
ax1.set_ylabel("Relative frequency [-]")
g, mean_z = comparison_histogram(uf_ch_list)
g.legend_.set_title(None)
legend = g.get_legend()
handles = legend.legendHandles
legend.remove()
g.legend(handles, ["observed point cloud", "resampled point cloud"], loc="upper right")
fig.savefig(plot_out_dir + "canopy_density_w_height.png")

mean_z[0] / mean_z[1]
