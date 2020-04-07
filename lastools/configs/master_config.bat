
:: folder containing lastools and license
SET DIR_LASTOOLS=C:\Users\Cob\index\educational\usask\research\masters\code_lib\lastools\LAStools\bin;
:: folder containing site polygons
SET SITE_MASK=C:\Users\Cob\index\educational\usask\research\masters\data\LiDAR\site_library\50_site_poly.shp
:: folder containing batch files
SET DIR_BAT=C:\Users\Cob\index\educational\usask\research\masters\repos\upper-clearing-lidar\lastools

:: PROJECTION
SET EPSG=32611

:: CLASSES
SET CLASS_GROUND=2
SET CLASS_VEGETATION=5
SET CLASS_NOISE=7

:: QUALIT CONTROL [1]
SET ORIGINAL_SCALE_FACTOR=0.00025
SET NUM_CORES=4
SET TILE_SIZE=25
SET TILE_BUFFER=5

:: CLASSIFICATION [2]
SET NOISE_ISOLATION=10
SET NOISE_STEP=1
SET GROUND_STEP=1.0
SET GROUND_OFFSET=.04

:: DEM [3]
SET DEM_RESOLUTION=1
SET DEM_THIN_RESOLUTION=.50
SET DEM_MAX_TIN_EDGE=3

:: CANOPY [4]
SET NOISE_HEIGHT_THRESHOLD_LOW=-1
SET NOISE_HEIGHT_THRESHOLD_HIGH=35
SET VEGETATION_HEIGHT_THRESHOLD_LOW=2
SET CHM_RESOLUTION=.25
SET CHM_MAX_TIN_EDGE=.75
SET CHM_LAYER_LIST= 0 2 5 10 15 20 25

SET BOOL_QC=0
SET BOOL_CLASSIFY=0
SET BOOL_DEM=1
SET BOOL_CHM=0
SET BOOL_COMPILE=0

:: "NOISE" <=> remove through step "NOISE"
SET RM_TEMP=NONE

SET DATE_LIST=19_149
:: 19_045 19_050 19_052 19_107 19_123 19_149

:: for just one date, copy and paste below:
:: call %DIR_BAT%\master_config_single_manual_patch.bat

for %%d in (%DATE_LIST%) do (

	SET PRODUCT_ID=%%d_all_200311
	:: folder in which temp and output files will be saved to
	SET DIR_WORKING=C:\Users\Cob\index\educational\usask\research\masters\data\LiDAR\%%d
	SET FILE_IN=C:\Users\Cob\index\educational\usask\research\masters\data\LiDAR\%%d\%%d_all_WGS84_utm11N.las

	call %DIR_BAT%\base_task_flow.bat
)