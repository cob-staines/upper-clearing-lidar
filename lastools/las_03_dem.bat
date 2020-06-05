:: las_03_output_dem.bat
:: dependencies
	:: CLASS_GROUND
    :: DEM_THIN_RESOLUTION
    :: DEM_RESOLUTION
    :: DEM_MAX_TIN_EDGE
    :: NUM_CORES
    :: EPSG

:: make output directories
mkdir .\TEMP_FILES\12_dem\res_%DEM_RESOLUTION%\
mkdir .\OUTPUT_FILES\DEM

:: build dem
blast2dem -i TEMP_FILES\08_classified\*.laz ^
        -keep_class %CLASS_GROUND% ^
        -use_tile_bb ^
        -thin_with_grid %DEM_THIN_RESOLUTION% ^
        -step %DEM_RESOLUTION% ^
        -kill %DEM_MAX_TIN_EDGE% ^
        -float_precision %ORIGINAL_SCALE_FACTOR% ^
        -ll %GRID_ORIGIN% ^
        -cores %NUM_CORES% ^
        -odir  TEMP_FILES\12_dem\res_%DEM_RESOLUTION%\ -obil -ocut 3 -odix _12

::lasgrid -i TEMP_FILES\10_dem\res_%DEM_RESOLUTION%\*.bil ^
::        -merged ^
::        -step %DEM_RESOLUTION% ^
::        -elevation ^
::        -nbits 32 ^
::        -epsg %EPSG% ^
::        -ll %GRID_ORIGIN% ^
::        -odir OUTPUT_FILES\DEM\ -obil -ocut 3 -odix dem_%DEM_RESOLUTION%m

:: merging of output products will now be done through the python workflow using gdal_merge.py