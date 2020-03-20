
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

SET GROUND_STEP_LIST= 0.125 0.25 0.5 1.0 2.0 4.0 8.0

:: DEM [3]
SET DEM_RESOLUTION=.04
SET DEM_THIN_RESOLUTION=.02
SET DEM_MAX_TIN_EDGE=.12

SET DATE=19_149

SET PRODUCT_ID=%DATE%_all_200311
:: folder in which temp and output files will be saved to
SET DIR_WORKING=C:\Users\Cob\index\educational\usask\research\masters\data\LiDAR\%DATE%
SET FILE_IN=C:\Users\Cob\index\educational\usask\research\masters\data\LiDAR\%DATE%\%DATE%_all_WGS84_utm11N.las

call %DIR_BAT%\las_00_dir_setup.bat

for %%d in (%GROUND_STEP_LIST%) do (

	::mkdir .\TEMP_FILES\06_ground_step_%%d
	:: ground classify tiles
	::lasground -i TEMP_FILES\05_noise\*.laz ^
    ::      	-step %%d ^
    ::      	-ignore_class %CLASS_NOISE% ^
    ::      	-cores %NUM_CORES% ^
    ::    	-odir TEMP_FILES\06_ground_step_%%d\ -olaz -ocut 3 -odix _06

    mkdir .\TEMP_FILES\10_dem\step_%%d
	mkdir .\OUTPUT_FILES\DEM

	:: build dem
	blast2dem -i TEMP_FILES\06_ground_step_%%d\*.laz ^
	        -keep_class %CLASS_GROUND% ^
	        -keep_last ^
	        -use_tile_bb ^
	        -thin_with_grid %DEM_THIN_RESOLUTION% ^
	        -step %DEM_RESOLUTION% ^
	        -kill %DEM_MAX_TIN_EDGE% ^
	        -cores %NUM_CORES% ^
	        -odir  TEMP_FILES\10_dem\step_%%d\ -obil -ocut 3 -odix _10

	lasgrid -i TEMP_FILES\10_dem\step_%%d\*.bil ^
            -merged ^
            -step %DEM_RESOLUTION% ^
            -epsg %EPSG% ^
            -odir OUTPUT_FILES\DEM\ -obil -ocut 3 -odix dem_%DEM_RESOLUTION%m_step_%%d


)

SET GROUND_STEP=0.25

mkdir .\TEMP_FILES\10_dem\res_%DEM_RESOLUTION%\step_0.125
mkdir .\OUTPUT_FILES\DEM

blast2dem -i TEMP_FILES\06_ground_step_0.125\*.laz ^
	        -keep_class %CLASS_GROUND% ^
	        -keep_last ^
	        -use_tile_bb ^
	        -thin_with_grid %DEM_THIN_RESOLUTION% ^
	        -step %DEM_RESOLUTION% ^
	        -kill %DEM_MAX_TIN_EDGE% ^
	        -cores %NUM_CORES% ^
	        -odir  TEMP_FILES\10_dem\res_%DEM_RESOLUTION%\step_0.125\ -obil -ocut 3 -odix _10

	lasgrid -i TEMP_FILES\10_dem\res_%DEM_RESOLUTION%\step_0.125\*.bil ^
            -merged ^
            -step %DEM_RESOLUTION% ^
            -epsg %EPSG% ^
            -odir OUTPUT_FILES\DEM\ -obil -ocut 3 -odix dem_%DEM_RESOLUTION%m_step_0.125
