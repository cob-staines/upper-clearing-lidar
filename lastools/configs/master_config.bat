
:: folder containing lastools and license
SET DIR_LASTOOLS=C:\Users\Cob\index\educational\usask\research\masters\code_lib\lastools\LAStools\bin;
:: folder containing site polygons
SET SITE_MASK=C:\Users\Cob\index\educational\usask\research\masters\data\LiDAR\site_library\site_poly.shp
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
SET NOISE_ISOLATION=20
SET NOISE_STEP=1.0
SET GROUND_STEP=2.0
SEt GROUND_CUTOFF_PERCENTILE=.4

:: DEM [3]
SET DEM_RESOLUTION=.25
SET DEM_THIN_RESOLUTION=.125
SET DEM_MAX_TIN_EDGE=.75


:: CANOPY [4]
SET NOISE_HEIGHT_THRESHOLD_LOW=-1
SET NOISE_HEIGHT_THRESHOLD_HIGH=40
SET VEGETATION_HEIGHT_THRESHOLD_LOW=2
SET CHM_RESOLUTION=.25
SET CHM_MAX_TIN_EDGE=.75
SET CHM_LAYER_LIST= 2 5 10 15 20 25 30 35

SET DATE_LIST= 19_149
:: 19_045 19_050 19_052 19_107 19_123 19_149

SET BOOL_QC=0
SET BOOL_CLASSIFY=0
SET BOOL_DEM=1
SET BOOL_CHM=1

for %%a in (%DATE_LIST%) do (

	SET PRODUCT_ID=%%a_all_test
		:: folder in which temp and output files will be saved to
	SET DIR_WORKING=C:\Users\Cob\index\educational\usask\research\masters\data\LiDAR\%%a
	SET FILE_IN=C:\Users\Cob\index\educational\usask\research\masters\data\LiDAR\%%a\%%a_all_WGS84_utm11N.las

	call %DIR_BAT%\base_task_flow.bat
)