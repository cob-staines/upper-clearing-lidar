:: folder in which temp and output files will be saved to
SET DIR_WORKING=C:\Users\Cob\index\educational\usask\research\masters\data\LiDAR\19_050
:: folder containing lastools and license
SET DIR_LASTOOLS=C:\Users\Cob\index\educational\usask\research\masters\code_lib\lastools\LAStools\bin;
:: folder containing site polygons
SET DIR_SITE_LIBRARY=C:\Users\Cob\index\educational\usask\research\masters\data\LiDAR\site_library


SET PRODUCT_ID=19_050_ladder_clearing_intensity-analysis
SET FILE_IN=C:\Users\Cob\index\educational\usask\research\masters\data\LiDAR\19_050\19_050_ladder_clearing_WGS84_utm11N.las

SET ORIGINAL_SCALE_FACTOR=0.00025
SET NUM_CORES=4
SET TILE_SIZE=100
SET TILE_BUFFER=20

:: initial setup
pushd %DIR_WORKING%

:: include LAStools in PATH to allow running script from here
set PATH=%PATH%;%DIR_LASTOOLS%

:: create clean folder for the output files of quality checking

mkdir .\TEMP_FILES
mkdir .\OUTPUT_FILES

cd TEMP_FILES

:: rmdir .\batch_out /s /q
mkdir .\01_precision
mkdir .\02_clip
mkdir .\03_tile
mkdir .\04_duplicate
mkdir .\05_noise
mkdir .\06_ground
mkdir .\07_vegetation
mkdir .\08_no_buffer

cd ..

:: ----------PROTOCOL---------- (sequential data processing)

:: should check each file prior to rescaling to verify actual precision
lasprecision -i %FILE_IN% ^
          -rescale %ORIGINAL_SCALE_FACTOR% %ORIGINAL_SCALE_FACTOR% %ORIGINAL_SCALE_FACTOR% ^
          -odir TEMP_FILES\01_precision\ -odix _01 -olaz

:: clip las by shpfile
lasclip -i TEMP_FILES\01_precision\%PRODUCT_ID%_01.laz ^
          -poly C:\Users\Cob\index\educational\usask\research\masters\data\LiDAR\site_library\site_poly.shp ^
          -odir TEMP_FILES\02_clip\ -ocut 3 -odix _02 -olaz

:: tile las for memory management
lastile -i TEMP_FILES\02_clip\%PRODUCT_ID%_02.laz ^
          -set_classification 0 -set_user_data 0 ^
          -tile_size 100 -buffer 20 ^
          -odir TEMP_FILES\03_tile\ -o %PRODUCT_ID%.laz

:: remove xyz duplicate points
lasduplicate -i TEMP_FILES\03_tile\*.laz ^
                -unique_xyz ^
                -cores 4 ^
                -odir TEMP_FILES\04_duplicate\ -olaz -odix _04

:: classify isolated points as class = 7
lasnoise -i TEMP_FILES\04_duplicate\*.laz ^
          -isolated 20 ^
          -step_xy 2.0 ^
          -step_z 2.0 ^
          -classify_as 7 ^
          -cores 4 ^
          -odir TEMP_FILES\05_noise\ -olaz -ocut 3 -odix _05

:: ground classify tiles
lasground -i TEMP_FILES\05_noise\*.laz ^
          -step 2 ^
          -ignore_class 7 ^
          -compute_height ^
          -cores 4 ^
          -odir TEMP_FILES\06_ground\ -olaz -ocut 3 -odix _06

lasclassify -i TEMP_FILES\06_ground\*.laz ^
          -ignore_class 2 7 ^
          -odir TEMP_FILES\07_vegetation\ -olaz -ocut 3 -odix _07

:: remove buffer
lastile -i TEMP_FILES\07_vegetation\*.laz ^
          -remove_buffer ^
          -cores 4 ^
          -odir TEMP_FILES\08_no_buffer\ -olaz -ocut 3 -odix _08

:: output ground points
lasmerge -i TEMP_FILES\08_no_buffer\*.laz ^
          -keep_class 2 ^
          -o OUTPUT_FILES\%PRODUCT_ID%_ground-points.laz

:: output high vegetation
lasmerge -i TEMP_FILES\08_no_buffer\*.laz ^
          -keep_class 5 ^
          -o OUTPUT_FILES\%PRODUCT_ID%_vegetation-points.laz

:: ladder_clearing
:: clip to upper clearing poly, filter by 1st return, filter to +/- 5 deg
lasclip -i OUTPUT_FILES\%PRODUCT_ID%_ground-points.laz ^
          -poly C:\Users\Cob\index\educational\usask\research\masters\data\LiDAR\site_library\upper_clearing_poly.shp ^
          -keep_single ^
          -keep_scan_angle -5 5 ^
          -o OUTPUT_FILES\%PRODUCT_ID%_clearing_ground-points_single-return_5deg.las