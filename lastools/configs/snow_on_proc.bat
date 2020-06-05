
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
SET SITE_MASK=C:\Users\Cob\index\educational\usask\research\masters\data\LiDAR\site_library\50_site_poly.shp


:: _____PARAMETER CONFIG_____
:: PROJECTION
SET EPSG=32611
SET GRID_ORIGIN=628004 5646470

:: CLASSES
SET CLASS_GROUND=2
SET CLASS_VEGETATION=5
SET CLASS_NOISE_GROUND=7
SET CLASS_NOISE_CANOPY=8


:: _____SET UP [0]_____
call %DIR_BAT%\las_00_dir_setup.bat


:: _____QUALIT CONTROL [1]_____
SET ORIGINAL_SCALE_FACTOR=0.00025
SET NUM_CORES=3
SET TILE_SIZE=75
SET TILE_BUFFER=5

call %DIR_BAT%\las_01_quality_control.bat


:: _____CLASSIFICATION [2]_____
SET NOISE_GROUND_ISOLATION=10
SET NOISE_GROUND_STEP=.33

:: ----optimized for snow-on DEM----
SET GROUND_STEP=1.0
SET GROUND_OFFSET=.03
:: --------

SET NOISE_CANOPY_ISOLATION=10
SET NOISE_CANOPY_STEP_XY=.5
SET NOISE_CANOPY_STEP_Z=.1

SET NOISE_HEIGHT_THRESHOLD_LOW=-1
SET NOISE_HEIGHT_THRESHOLD_HIGH=35
SET VEGETATION_HEIGHT_THRESHOLD_LOW=2

call %DIR_BAT%\las_02_classification.bat


:: _____DEM [3]_____
SET DEM_RESOLUTION=.04
SET DEM_THIN_RESOLUTION=.02
SET DEM_MAX_TIN_EDGE=.12
call %DIR_BAT%\las_03_dem.bat

SET DEM_RESOLUTION=.10
SET DEM_THIN_RESOLUTION=.05
SET DEM_MAX_TIN_EDGE=.30
call %DIR_BAT%\las_03_dem.bat

SET DEM_RESOLUTION=.25
SET DEM_THIN_RESOLUTION=.125
SET DEM_MAX_TIN_EDGE=.75
call %DIR_BAT%\las_03_dem.bat

SET DEM_RESOLUTION=.50
SET DEM_THIN_RESOLUTION=.25
SET DEM_MAX_TIN_EDGE=1.50
call %DIR_BAT%\las_03_dem.bat

SET DEM_RESOLUTION=1.00
SET DEM_THIN_RESOLUTION=.50
SET DEM_MAX_TIN_EDGE=3.00
call %DIR_BAT%\las_03_dem.bat

:: _____CANOPY [4]_____
SET CHM_RESOLUTION=.10
SET CHM_MAX_TIN_EDGE=.30
call %DIR_BAT%\las_04_canopy.bat

SET CHM_RESOLUTION=.25
SET CHM_MAX_TIN_EDGE=.75
call %DIR_BAT%\las_04_canopy.bat


:: _____COMPILE [5]_____
call %DIR_BAT%\las_05_compile.bat

:: _____RASTER PRODUCTS [6]_____
::SET RAS_RESOLUTION=1
::call %DIR_BAT%\las_06_raster_products.bat