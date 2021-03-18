import subprocess
import rastools

lastools_dir = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\code_lib\\lastools\\LAStools\\bin\\"

def mCH(las_file):
    norm_file = las_file.replace(".las", "_normalized.las")
    mch_file = las_file.replace(".las", "_mCH.bil")

    epsg = '32611'
    res = str(0.10)
    gp_file = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\LAS\\19_149_las_proc_ground_thinned_merged.las"

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

    # calculate mean canopy height
    lasgrid_cmd = [lastools_dir + "lasgrid.exe",
                '-i', norm_file,
                '-epsg', epsg,
                '-keep_z', z_min, z_max,
                '-step', res,
                '-elevation',
                '-mean',
                '-o', mch_file]

    pcs = subprocess.Popen(lasgrid_cmd)  # start process
    pcs.wait()  # wait for it to finish

    ras = rastools.raster_load(mch_file)
    ras.data[ras.data == ras.no_data] = 0
    rastools.raster_save(ras, mch_file, file_format="EHdr")

las_in = ["C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\ray_sampling\\sources\\045_050_052_combined_WGS84_utm11N_r0.25_vox_resampled.las",
          r"C:\Users\Cob\index\educational\usask\research\masters\data\lidar\19_149\19_149_las_proc\OUTPUT_FILES\LAS\19_149_las_proc_classified_merged.las",
          r"C:\Users\Cob\index\educational\usask\research\masters\data\lidar\19_149\19_149_las_proc\OUTPUT_FILES\LAS\19_149_las_proc_classified_merged_poisson_0.15.las"]

for ff in las_in:
    mCH(ff)