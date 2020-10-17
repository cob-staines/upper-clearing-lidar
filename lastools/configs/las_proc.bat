
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


:: _____QUALIT CONTROL [1]_____
SET ORIGINAL_SCALE_FACTOR=0.00025
SET NUM_CORES=3
SET TILE_SIZE=50
SET TILE_BUFFER=2

call %DIR_BAT%\las_01_quality_control.bat


:: _____GROUND CLASSIFICATION [2]_____
SET GROUND_STEP=.5
SET GROUND_OFFSET=.025
SET GROUND_SPIKE=.05
SET GROUND_THIN_STEP=.03
SET GROUND_THIN_PERCENTILE=50

call %DIR_BAT%\las_02_ground_classification.bat


:::: _____DEM [3]_____
::SET DEM_RESOLUTION=.05
::SET DEM_MAX_TIN_EDGE=.15
::call %DIR_BAT%\las_03_dem.bat
::
::SET DEM_RESOLUTION=.10
::SET DEM_MAX_TIN_EDGE=.30
::call %DIR_BAT%\las_03_dem.bat
::
::SET DEM_RESOLUTION=.25
::SET DEM_MAX_TIN_EDGE=.75
::call %DIR_BAT%\las_03_dem.bat
::
::SET DEM_RESOLUTION=1.00
::SET DEM_MAX_TIN_EDGE=3.00
::call %DIR_BAT%\las_03_dem.bat


:::: _____CANOPY CLASSIFICATION [4]_____
::SET NOISE_CANOPY_ISOLATION=10
::SET NOISE_CANOPY_STEP_XY=.5
::SET NOISE_CANOPY_STEP_Z=.1
::
::SET NOISE_HEIGHT_THRESHOLD_LOW=-.05
::SET NOISE_HEIGHT_THRESHOLD_HIGH=35
::SET VEGETATION_HEIGHT_THRESHOLD_LOW=.5
::
::call %DIR_BAT%\las_04_canopy_classification.bat

::
:: _____CHM [5]_____
::SET CHM_RESOLUTION=.10
::SET CHM_MAX_TIN_EDGE=.30
::call %DIR_BAT%\las_05_chm.bat
::
::SET CHM_RESOLUTION=.25
::SET CHM_MAX_TIN_EDGE=.75
::call %DIR_BAT%\las_05_chm.bat

::
:: _____RASTER PRODUCTS [6]_____
::SET RAS_RESOLUTION=.10
::
::SET RAS_MAX_ANGLE=5
::call %DIR_BAT%\las_06_raster_products.bat
::SET RAS_MAX_ANGLE=10
::call %DIR_BAT%\las_06_raster_products.bat
::SET RAS_MAX_ANGLE=15
::call %DIR_BAT%\las_06_raster_products.bat
::SET RAS_MAX_ANGLE=30
::call %DIR_BAT%\las_06_raster_products.bat

:: _____RASTER TEMPLATES [7]_____
::SET RAS_RESOLUTION=.05
::call %DIR_BAT%\las_07_raster_templates.bat
::SET RAS_RESOLUTION=.10
::call %DIR_BAT%\las_07_raster_templates.bat
:;SET RAS_RESOLUTION=.25
::call %DIR_BAT%\las_07_raster_templates.bat
::SET RAS_RESOLUTION=1.00
::call %DIR_BAT%\las_07_raster_templates.bat

:: _____COMPILE LAS [8]_____
::call %DIR_BAT%\las_08_compile.bat