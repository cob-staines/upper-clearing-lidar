SET PRODUCT_ID=19_149_gridded_120m_intensity-analysis
SET FILE_IN=C:\Users\Cob\index\educational\usask\research\masters\data\LiDAR\19_149\19_149_gridded_120m_WGS84_utm11N.las

:: folder in which temp and output files will be saved to
SET DIR_WORKING=C:\Users\Cob\index\educational\usask\research\masters\data\LiDAR\19_149
:: folder containing lastools and license
SET DIR_LASTOOLS=C:\Users\Cob\index\educational\usask\research\masters\code_lib\lastools\LAStools\bin;
:: folder containing site polygons
SET DIR_SITE_LIBRARY=C:\Users\Cob\index\educational\usask\research\masters\data\LiDAR\site_library
:: folder containing batch files
SET DIR_BAT=C:\Users\Cob\index\educational\usask\research\masters\repos\upper-clearing-lidar\lastools

:: PROJECTION
SET EPSG=32611

:: CLASSES
SET CLASS_GROUND=2
SET CLASS_NOISE=7


:: __________ PROTOCOL__________

SET ORIGINAL_SCALE_FACTOR=0.00025
SET NUM_CORES=4
SET TILE_SIZE=100
SET TILE_BUFFER=20

call %DIR_BAT%\las_01_quality_control.bat
:: dependencies
     :: DIR_WORKING
     :: DIR_LASTOOLS
     :: DIR_SITE_LIBRARY
     :: PRODUCT_ID
     :: FILE_IN
     :: ORIGINAL_SCALE_FACTOR
     :: NUM_CORES
     :: TILE_SIZE
     :: TILE_BUFFER

SET NOISE_ISOLATION=20
SET NOISE_STEP=2.0
SET GROUND_STEP=2.0
SET HEIGHT_THRESHOLD_LOW=-3
SET HEIGHT_THRESHOLD_HIGH=40


call %DIR_BAT%\las_02_classification.bat
:: dependencies
     :: CLASS_GROUND
     :: CLASS_NOISE
     :: NUM_CORES
     :: NOISE_ISOLATION
     :: NOISE_STEP
     :: GROUND_STEP
     :: HEIGHT_THRESHOLD_LOW
     :: HEIGHT_THRESHOLD_HIGH

call %DIR_BAT%\las_03_remove_buffer.bat
:: dependencies
     :: NUM_CORES

SET RESOLUTION_DEM=.25
SET RESOLUTION_THIN=.125
SET MAX_TIN_EDGE=1
call %DIR_BAT%\las_03_output_dem.bat
:: dependencies
     :: CLASS_GROUND
     :: NUM_CORES
     :: RESOLUTION_DEM
     :: RESOLUTION_THIN
     :: MAX_TIN_EDGE

:: __________ MANUAL OUTPUTS __________

mkdir .\OUTPUT_FILES

:: output ground points
lasmerge -i TEMP_FILES\09_no_buffer\*.laz ^
          -keep_class 2 ^
          -o OUTPUT_FILES\%PRODUCT_ID%_ground-points.laz

:: output high vegetation
lasmerge -i TEMP_FILES\09_no_buffer\*.laz ^
          -keep_class 5 ^
          -o OUTPUT_FILES\%PRODUCT_ID%_vegetation-points.laz

:: ladder_clearing
:: clip to upper clearing poly, filter by 1st return, filter to +/- 5 deg
lasclip -i OUTPUT_FILES\%PRODUCT_ID%_ground-points.laz ^
          -poly C:\Users\Cob\index\educational\usask\research\masters\data\LiDAR\site_library\upper_clearing_poly.shp ^
          -keep_single ^
          -keep_scan_angle -5 5 ^
          -o OUTPUT_FILES\%PRODUCT_ID%_clearing_ground-points_single-return_5deg.las

lasgrid -i OUTPUT_FILES\%PRODUCT_ID%_ground-points.laz ^
          -step 0.25 ^
          -point_density ^
          -o OUTPUT_FILES\%PRODUCT_ID%_point-density.bil

lasgrid -i OUTPUT_FILES\%PRODUCT_ID%_ground-points.laz ^
          -step 0.25 ^
          -elevation -stddev ^
          -o OUTPUT_FILES\%PRODUCT_ID%_elevation-stddev.bil

lasgrid -i OUTPUT_FILES\%PRODUCT_ID%_ground-points.laz ^
          -step 0.25 ^
          -intensity -mean ^
          -o OUTPUT_FILES\%PRODUCT_ID%_intentisy-mean.bil
          
lasgrid -i OUTPUT_FILES\%PRODUCT_ID%_ground-points.laz ^
          -step 0.25 ^
          -scan_angle_abs_lowest ^
          -o OUTPUT_FILES\%PRODUCT_ID%_scan_angle_abs_lowest.bil
