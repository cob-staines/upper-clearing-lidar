:: las_04_canopy_raster.bat
:: dependencies
    :: CHM_RESOLUTION
    :: CHM_MAX_TIN_EDGE
    :: CLASS_VEGETATION
    :: NUM_CORES
    :: EPSG

:: make output directories
mkdir .\TEMP_FILES\12_chm\res_%CHM_RESOLUTION%\
mkdir .\OUTPUT_FILES\CHM


:: pit free CHM following Khosravipour et al. 2016, 2015, 2014

las2dem -i TEMP_FILES\09_normalized\*.laz ^
      -use_tile_bb ^
      -drop_class %CLASS_NOISE_GROUND% %CLASS_NOISE_CANOPY% ^
      -spike_free %CHM_MAX_TIN_EDGE% ^
      -step %CHM_RESOLUTION% ^
      -kill %CHM_MAX_TIN_EDGE% ^
      -ll %GRID_ORIGIN% ^
      -cores %NUM_CORES% ^
      -odir TEMP_FILES\12_chm\res_%CHM_RESOLUTION%\ -ocut 3 -odix _spike_free_chm_%CHM_RESOLUTION%m -obil

lasgrid -i TEMP_FILES\12_chm\res_%CHM_RESOLUTION%\*.bil ^
      -merged ^
      -step %CHM_RESOLUTION% ^
      -epsg %EPSG% ^
      -ll %GRID_ORIGIN% ^
      -odir OUTPUT_FILES\CHM\ -obil