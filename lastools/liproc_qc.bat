
:: config
SET DIR_WORKING=C:\Users\Cob\index\educational\usask\research\masters\data\LiDAR\19_149
SET DIR_LASTOOLS=C:\Users\Cob\index\educational\usask\research\masters\code_lib\lastools\LAStools\bin;
SET DIR_SITE_LIBRARY=C:\Users\Cob\index\educational\usask\research\masters\data\LiDAR\site_library

SET FILE_IN_BASE=19_149_all_WGS84_utm11N_nocolor
SET FILE_IN_EXT=.las

SET ORIGINAL_SCALE_FACTOR=0.00025

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
mkdir .\07_no_buffer
mkdir .\08_exports

cd ..

:: ----------PROTOCOL---------- (sequential data processing)

:: should check each file prior to rescaling to verify actual precision
lasprecision -i %FILE_IN_BASE%%FILE_IN_EXT% ^
          -rescale %ORIGINAL_SCALE_FACTOR% %ORIGINAL_SCALE_FACTOR% %ORIGINAL_SCALE_FACTOR% ^
          -odir TEMP_FILES\01_precision\ -odix _01 -olaz

:: clip las by shpfile
lasclip -i TEMP_FILES\01_precision\%FILE_IN_BASE%_01.laz ^
          -poly C:\Users\Cob\index\educational\usask\research\masters\data\LiDAR\site_library\site_poly.shp ^
          -odir TEMP_FILES\02_clip\ -ocut 3 -odix _02 -olaz

:: tile las for memory management
lastile -i TEMP_FILES\02_clip\%FILE_IN_BASE%_02.laz ^
          -set_classification 0 -set_user_data 0 ^
          -tile_size 100 -buffer 20 ^
          -odir TEMP_FILES\03_tile\ -o %FILE_IN_BASE%.laz

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

:: remove buffer
lastile -i batch_out\06_ground\*.laz ^
          -remove_buffer ^
          -cores 4 ^
          -odir batch_out\07_no_buffer\ -olaz -ocut 2 -odix 07

:: merge all ground points into one file
lasmerge -i batch_out\07_no_buffer\*.laz ^
          -keep_class 2 ^
          -o batch_out\07_no_buffer\19_149_all_WGS84_utm11_ground-points.laz

:: clip to upper clearing poly
lasclip -i batch_out\07_no_buffer\19_149_all_WGS84_utm11_ground-points.laz ^
          -poly batch_out\upper_clearing_poly.shp ^
          -o batch_out\07_no_buffer\19_149_all_WGS84_utm11_ground-points_upper-clearing.las

:: clip to upper forest poly
lasclip -i batch_out\07_no_buffer\19_149_all_WGS84_utm11_ground-points.laz ^
          -poly batch_out\upper_forest_poly.shp ^
          -o batch_out\07_no_buffer\19_149_all_WGS84_utm11_ground-points_upper-forest.las

:: output raster of point density
lasgrid -i batch_out\07_no_buffer\19_149_all_WGS84_utm11_ground-points.laz ^
          -point_density ^
          -step 0.25 ^
          -nbits 16 ^
          -o batch_out\08_exports\ground_point_density.png
