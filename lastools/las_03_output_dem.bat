:: las_03_output_dem.bat
:: dependencies
	:: CLASS_GROUND
    :: NUM_CORES
    :: RESOLUTION_DEM
    :: RESOLUTION_THIN
    :: MAX_TIN_EDGE

cd TEMP_FILES
mkdir .\10_dem
cd ..

mkdir OUTPUT_FILES

cd OUTPUT_FILES
mkdir .\DEM
cd ..

:: build dem

las2dem -i TEMP_FILES\08_vegetation\*.laz ^
        -keep_class %CLASS_GROUND% ^
        -thin_with_grid %RESOLUTION_THIN% -step %RESOLUTION_DEM% ^
        -use_tile_bb ^
        -kill %MAX_TIN_EDGE% ^
        -cores %NUM_CORES% ^
        -odir  TEMP_FILES\10_dem\ -obil -ocut 3 -odix _10

lasgrid -i TEMP_FILES\10_dem\*.bil ^
            -merged ^
            -step %RESOLUTION_DEM% ^
            -epsg %EPSG% ^
            -odir OUTPUT_FILES\DEM\ -obil -ocut 3 -odix single_dem