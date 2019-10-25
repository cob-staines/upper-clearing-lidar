
:: one approach to create a huge seamless hillshaded DTM of a densely
:: forested area and steep terrain as archeologists often want to do
::

:: include LAStools in PATH to allow running script from here

cd C:\Users\Cob\index\educational\usask\research\masters\data\LiDAR\19_149

set PATH=%PATH%;C:\Users\Cob\index\educational\usask\research\masters\code_lib\lastools\LAStools\bin;

:: create clean folder for the output files of quality checking

:: rmdir .\batch_out /s /q
mkdir .\batch_out
mkdir .\batch_out\01_precision
mkdir .\batch_out\02_clip
mkdir .\batch_out\03_tile
mkdir .\batch_out\04_duplicate
mkdir .\batch_out\05_noise
mkdir .\batch_out\06_ground
mkdir .\batch_out\07_no_buffer
mkdir .\batch_out\08_exports

:: ----------PROTOCOL---------- (sequential data processing)

:: should check each file prior to rescaling to verify actual precision
lasprecision -i 19_149_all_WGS84_utm11N_nocolor.las ^
          -rescale 0.00025 0.00025 0.00025 ^
          -o batch_out\01_precision\19_149_all_WGS84_utm11_01.laz

:: clip las by shpfile
lasclip -i batch_out\01_precision\19_149_all_WGS84_utm11_01.laz ^
          -poly batch_out\site_poly.shp ^
          -o batch_out\02_clip\19_149_all_WGS84_utm11_02.laz

:: tile las for memory management
lastile -i batch_out\02_clip\19_149_all_WGS84_utm11_02.laz ^
          -set_classification 0 -set_user_data 0 ^
          -tile_size 100 -buffer 20 ^
          -o batch_out\03_tile\19_149_all_WGS84_utm11.laz

:: remove xyz duplicate points
lasduplicate -i batch_out\03_tile\*.laz ^
                -unique_xyz ^
                -cores 4 ^
                -odir batch_out\04_duplicate\ -olaz -odix _04

:: classify isolated points as class = 7
lasnoise -i batch_out\04_duplicate\*.laz ^
          -isolated 20 ^
          -step_xy 2.0 ^
          -step_z 2.0 ^
          -classify_as 7 ^
          -cores 4 ^
          -odir batch_out\05_noise\ -olaz -ocut 2 -odix 05

:: ground classify tiles
lasground -i batch_out\05_noise\*.laz ^
          -step 2 ^
          -ignore_class 7 ^
          -compute_height ^
          -cores 4 ^
          -odir batch_out\06_ground\ -olaz -ocut 2 -odix 06

:: remove buffer
lastile -i batch_out\06_ground\*.laz ^
          -remove_buffer ^
          -cores 4 ^
          -odir batch_out\07_no_buffer\ -olaz -ocut 2 -odix 07

:: merge all ground points into one file
lasmerge -i batch_out\07_no_buffer\*.laz ^
          -keep_class 2 ^
          -o batch_out\07_no_buffer\19_149_all_WGS84_utm11_ground-points.laz

:: clip to upper clearing
lasclip -i batch_out\07_no_buffer\19_149_all_WGS84_utm11_ground-points.laz ^
          -poly batch_out\upper_clearing_poly.shp ^
          -o batch_out\07_no_buffer\19_149_all_WGS84_utm11_ground-points_upper-clearing.laz

:: clip to upper forest
lasclip -i batch_out\07_no_buffer\19_149_all_WGS84_utm11_ground-points.laz ^
          -poly batch_out\upper_forest_poly.shp ^
          -o batch_out\07_no_buffer\19_149_all_WGS84_utm11_ground-points_upper-forest.laz

:: output raster of point density
lasgrid -i batch_out\07_no_buffer\19_149_all_WGS84_utm11_ground-points.laz ^
          -point_density ^
          -step 0.25 ^
          -nbits 16 ^
          -o batch_out\08_exports\ground_point_density.png

:: ---------END PROTOCOL----------

las2tin -i batch_out\t_05_ground.laz ^
          -keep_class 2 ^
          -kill 1 ^
          -o batch_out\t_06_tin.shp

:: create one seamlessly hillshaded DTM with BLAST from the ground
blast2dem -i batch_out\ground_01.laz ^
          –keep_class 2 ^
          -step 1.0 ^
          -kill 3 ^
          -o batch_out\dem_01.png

:: ----------TOOLBOX---------- (no particula order here)

:: output raster of point density
lasgrid -i batch_out\06_ground\*laz ^
          -keep_class 2 ^
          -point_density ^
          -step 1 ^
          -use_tile_bb ^
          -nbits 16
          -odir batch_out\07_gpd\ -opng

:: determine original scale factor
lasprecision -i 19_149_all_WGS84_utm11N_nocolor.las -all

:: determine difference between two las/laz files
lasdiff -i 19_149_all_WGS84_utm11N_nocolor.las -i batch_out\precision_01.laz