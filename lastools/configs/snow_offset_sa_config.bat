
:: _____PROJECT CONFIG_____
SET DATE=19_045
SET PRODUCT_ID=%DATE%_snow_on


:: _____DIRECTORY CONFIG_____
:: parent dir of project folder
SET DIR_WORKING=C:\Users\Cob\index\educational\usask\research\masters\data\LiDAR\%DATE%
:: initial las file
SET FILE_IN=C:\Users\Cob\index\educational\usask\research\masters\data\LiDAR\%DATE%\%DATE%_all_WGS84_utm11N.las
:: folder containing batch files
SET DIR_BAT=C:\Users\Cob\index\educational\usask\research\masters\repos\upper-clearing-lidar\lastools
:: folder containing lastools and license
SET DIR_LASTOOLS=C:\Users\Cob\index\educational\usask\research\masters\code_lib\lastools\LAStools\bin;
:: folder containing site polygons
SET SITE_MASK=C:\Users\Cob\index\educational\usask\research\masters\data\LiDAR\site_library\50_site_poly.shp


:: _____PARAMETER CONFIG_____
:: PROJECTION
SET EPSG=32611
SET GRID_ORIGIN=628004 5646470

:: CLASSES
SET CLASS_GROUND=2
SET CLASS_VEGETATION=5
SET CLASS_NOISE=7


:: _____SET UP [0]_____
call %DIR_BAT%\las_00_dir_setup.bat


:: _____QUALIT CONTROL [1]_____
SET ORIGINAL_SCALE_FACTOR=0.00025
SET NUM_CORES=4
SET TILE_SIZE=25
SET TILE_BUFFER=5


:: _____CLASSIFICATION [2]_____
SET NOISE_ISOLATION=10
SET NOISE_STEP=1

:: _____DEM [3]_____
SET DEM_RESOLUTION=.04
SET DEM_THIN_RESOLUTION=.02
SET DEM_MAX_TIN_EDGE=.12

:: optimization parameters
SET GROUND_STEP=.5
SET GROUND_OFFSET=.01
call %DIR_BAT%\step_offset_opt.bat

:: optimization parameters
SET GROUND_STEP=.5
SET GROUND_OFFSET=.03
call %DIR_BAT%\step_offset_opt.bat

:: optimization parameters
SET GROUND_STEP=.5
SET GROUND_OFFSET=.04
call %DIR_BAT%\step_offset_opt.bat

:: optimization parameters
SET GROUND_STEP=.5
SET GROUND_OFFSET=.05
call %DIR_BAT%\step_offset_opt.bat

:: optimization parameters
SET GROUND_STEP=1
SET GROUND_OFFSET=.01
call %DIR_BAT%\step_offset_opt.bat

:: optimization parameters
SET GROUND_STEP=1
SET GROUND_OFFSET=.03
call %DIR_BAT%\step_offset_opt.bat

:: optimization parameters
SET GROUND_STEP=1
SET GROUND_OFFSET=.04
call %DIR_BAT%\step_offset_opt.bat

:: optimization parameters
SET GROUND_STEP=1
SET GROUND_OFFSET=.05
call %DIR_BAT%\step_offset_opt.bat

:: optimization parameters
SET GROUND_STEP=2
SET GROUND_OFFSET=.01
call %DIR_BAT%\step_offset_opt.bat

:: optimization parameters
SET GROUND_STEP=2
SET GROUND_OFFSET=.03
call %DIR_BAT%\step_offset_opt.bat

:: optimization parameters
SET GROUND_STEP=2
SET GROUND_OFFSET=.04
call %DIR_BAT%\step_offset_opt.bat

:: optimization parameters
SET GROUND_STEP=2
SET GROUND_OFFSET=.05
call %DIR_BAT%\step_offset_opt.bat
