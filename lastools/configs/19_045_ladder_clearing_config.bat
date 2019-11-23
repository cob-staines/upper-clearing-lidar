SET PRODUCT_ID=19_045_ladder_clearing_intensity-analysis
SET FILE_IN=C:\Users\Cob\index\educational\usask\research\masters\data\LiDAR\19_045\19_045_ladder_clearing_WGS84_utm11N.las

:: folder in which temp and output files will be saved to
SET DIR_WORKING=C:\Users\Cob\index\educational\usask\research\masters\data\LiDAR\19_045
:: folder containing lastools and license
SET DIR_LASTOOLS=C:\Users\Cob\index\educational\usask\research\masters\code_lib\lastools\LAStools\bin;
:: shp file for site mask
SET SITE_MASK=C:\Users\Cob\index\educational\usask\research\masters\data\LiDAR\site_library\site_poly.shp
:: folder containing batch files
SET DIR_BAT=C:\Users\Cob\index\educational\usask\research\masters\repos\upper-clearing-lidar\lastools

:: PROJECTION
SET EPSG=32611

:: CLASSES
SET CLASS_GROUND=2
SET CLASS_NOISE=7


call %DIR_BAT%\las_00_dir_setup.bat

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
     :: NUM_CORES
     :: NOISE_ISOLATION
     :: NOISE_STEP
     :: GROUND_STEP

call %DIR_BAT%\las_03_remove_buffer.bat
:: dependencies
     :: NUM_CORES


:: __________ MANUAL OUTPUTS __________

mkdir .\OUTPUT_FILES

:: output ground points
lasmerge -i TEMP_FILES\08_no_buffer\*.laz ^
          -keep_class 2 ^
          -o OUTPUT_FILES\%PRODUCT_ID%_ground-points.laz

:: output high vegetation
lasmerge -i TEMP_FILES\08_no_buffer\*.laz ^
          -keep_class 5 ^
          -o OUTPUT_FILES\%PRODUCT_ID%_vegetation-points.las

:: ladder_clearing
:: clip to upper clearing poly, filter by 1st return, filter to +/- 5 deg
lasclip -i OUTPUT_FILES\%PRODUCT_ID%_ground-points.laz ^
          -poly C:\Users\Cob\index\educational\usask\research\masters\data\LiDAR\site_library\upper_clearing_poly.shp ^
          -keep_single ^
          -keep_scan_angle -5 5 ^
          -o OUTPUT_FILES\%PRODUCT_ID%_clearing_ground-points_single-return_5deg.las

:: output all non-noise
lasmerge -i TEMP_FILES\08_no_buffer\*.laz ^
          -drop_class 7 ^
          -o OUTPUT_FILES\%PRODUCT_ID%_all-points.las



:: output high vegetation first returns above height

lasheight -i TEMP_FILES\06_ground\*.laz ^
          -classify_above 15 5 -classify_above 40 7 ^
          -cores %NUM_CORES% ^
          -odir TEMP_FILES\07_vegetation\ -olaz -ocut 3 -odix _07

lasmerge -i TEMP_FILES\08_no_buffer\*.laz ^
          -keep_class 5 ^
          -keep_first ^
          -o OUTPUT_FILES\%PRODUCT_ID%_vegetation-points_above-15m.las
