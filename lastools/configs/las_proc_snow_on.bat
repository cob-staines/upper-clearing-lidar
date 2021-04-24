
:: _____DIRECTORY CONFIG_____

:: folder containing batch files
SET DIR_BAT=C:\Users\Cob\index\educational\usask\research\masters\repos\upper-clearing-lidar\lastools
:: folder containing lastools and license
SET DIR_LASTOOLS=C:\Users\Cob\index\educational\usask\research\masters\code_lib\lastools\LAStools\bin;
:: parent dir of project folder
SET DIR_WORKING=C:\Users\Cob\index\educational\usask\research\masters\data\LiDAR\%DATE%
:: initial las file
SET FILE_IN=C:\Users\Cob\index\educational\usask\research\masters\data\LiDAR\%DATE%\%DATE%_all_WGS84_utm11N.las
:: site polygon
SET SITE_MASK=C:\Users\Cob\index\educational\usask\research\masters\data\LiDAR\site_library\mb_65_poly.shp

SET NUM_CORES=3

:: _____PARAMETER CONFIG_____
:: PROJECTION
SET EPSG=32611
:: SET GRID_ORIGIN=628004 5646470  (50_site)
SET GRID_ORIGIN=627991 5646454

:: CLASSES
SET CLASS_GROUND=2
SET CLASS_VEGETATION=5
SET CLASS_NOISE=7


:: _____SET UP [0]_____
call %DIR_BAT%\las_00_dir_setup.bat


:: _____QUALITY CONTROL [1]_____
SET ORIGINAL_SCALE_FACTOR=0.00025
SET TILE_SIZE=50
SET TILE_BUFFER=2

:: call %DIR_BAT%\las_01_quality_control.bat


:: _____GROUND CLASSIFICATION [2]_____
SET GROUND_STEP=.5
SET GROUND_OFFSET=.1
SET GROUND_SPIKE=.1
SET GROUND_THIN_STEP=.05
SET GROUND_THIN_PERCENTILE=50

:: call %DIR_BAT%\las_02_ground_classification.bat


:: _____DEM [3]_____
SET DEM_RESOLUTION=.05
SET DEM_MAX_TIN_EDGE=.15
:: call %DIR_BAT%\las_03_dem.bat

SET DEM_RESOLUTION=.10
SET DEM_MAX_TIN_EDGE=.30
:: call %DIR_BAT%\las_03_dem.bat

SET DEM_RESOLUTION=.25
SET DEM_MAX_TIN_EDGE=.75
:: call %DIR_BAT%\las_03_dem.bat

SET DEM_RESOLUTION=1.00
SET DEM_MAX_TIN_EDGE=3.00
:: call %DIR_BAT%\las_03_dem.bat


:: _____HS NORMALIZTION [9]_____
SET GROUND_POINTS_FILE=C:\Users\Cob\index\educational\usask\research\masters\data\lidar\19_149\19_149_las_proc\OUTPUT_FILES\LAS\19_149_las_proc_ground_thinned_merged.las
:: call %DIR_BAT%\las_09_snow_depth_normalization.bat

:: _____HS DSM [10]_____
SET INTERP_LEN=0
SET DSM_RESOLUTION=.05
SET DSM_MAX_TIN_EDGE=1000
:: call %DIR_BAT%\las_10_snow_dsm.bat

SET DSM_RESOLUTION=.10
SET DSM_MAX_TIN_EDGE=1000
:: call %DIR_BAT%\las_10_snow_dsm.bat

SET DSM_RESOLUTION=.25
SET DSM_MAX_TIN_EDGE=1000
:: call %DIR_BAT%\las_10_snow_dsm.bat

SET DSM_RESOLUTION=1.00
SET DSM_MAX_TIN_EDGE=1000
:: call %DIR_BAT%\las_10_snow_dsm.bat

SET INTERP_LEN=1
SET DSM_RESOLUTION=.05
SET DSM_MAX_TIN_EDGE=.05
:: call %DIR_BAT%\las_10_snow_dsm.bat

SET DSM_RESOLUTION=.10
SET DSM_MAX_TIN_EDGE=.10
:: call %DIR_BAT%\las_10_snow_dsm.bat

SET DSM_RESOLUTION=.25
SET DSM_MAX_TIN_EDGE=.25
:: call %DIR_BAT%\las_10_snow_dsm.bat

SET DSM_RESOLUTION=1.00
SET DSM_MAX_TIN_EDGE=1.00
:: call %DIR_BAT%\las_10_snow_dsm.bat


SET INTERP_LEN=2
SET DSM_RESOLUTION=.05
SET DSM_MAX_TIN_EDGE=.10
:: call %DIR_BAT%\las_10_snow_dsm.bat

SET DSM_RESOLUTION=.10
SET DSM_MAX_TIN_EDGE=.20
:: call %DIR_BAT%\las_10_snow_dsm.bat

SET DSM_RESOLUTION=.25
SET DSM_MAX_TIN_EDGE=.50
:: call %DIR_BAT%\las_10_snow_dsm.bat

SET DSM_RESOLUTION=1.00
SET DSM_MAX_TIN_EDGE=2.00
:: call %DIR_BAT%\las_10_snow_dsm.bat


SET INTERP_LEN=3
SET DSM_RESOLUTION=.05
SET DSM_MAX_TIN_EDGE=.15
:: call %DIR_BAT%\las_10_snow_dsm.bat

SET DSM_RESOLUTION=.10
SET DSM_MAX_TIN_EDGE=.30
:: call %DIR_BAT%\las_10_snow_dsm.bat

SET DSM_RESOLUTION=.25
SET DSM_MAX_TIN_EDGE=.75
:: call %DIR_BAT%\las_10_snow_dsm.bat

SET DSM_RESOLUTION=1.00
SET DSM_MAX_TIN_EDGE=3.00
:: call %DIR_BAT%\las_10_snow_dsm.bat


::_____COMPILE [6]_____
:: call %DIR_BAT%\las_06_remove_buffer.bat

:: _____RASTER PRODUCTS [7]_____
SET RAS_RESOLUTION=.10
:: call %DIR_BAT%\las_07_raster_ground_point_density.bat
SET RAS_RESOLUTION=.25
:: call %DIR_BAT%\las_07_raster_ground_point_density.bat