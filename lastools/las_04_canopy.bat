:: las_04_canopy_raster.bat
:: dependencies
    :: CHM_RESOLUTION
    :: CHM_MAX_TIN_EDGE
    :: CLASS_VEGETATION
    :: NUM_CORES
    :: EPSG

:: make output directories
mkdir .\TEMP_FILES\12_chm\res_%CHM_RESOLUTION%\merged
mkdir .\OUTPUT_FILES\CHM


:: pit free CHM following Khosravipour et al. 2014

for %%a in (%CHM_LAYER_LIST%) do (

    mkdir .\TEMP_FILES\12_chm\res_%CHM_RESOLUTION%\h_%%a

    blast2dem -i TEMP_FILES\09_normalized\*.laz ^
          -drop_class %CLASS_NOISE_GROUND% %CLASS_NOISE_CANOPY% ^
          -keep_first ^
          -drop_z_below %%a ^
          -use_tile_bb ^
          -step %CHM_RESOLUTION% ^
          -kill %CHM_MAX_TIN_EDGE% ^
          -ll %GRID_ORIGIN% ^
          -cores %NUM_CORES% ^
          -odir TEMP_FILES\12_chm\res_%CHM_RESOLUTION%\h_%%a\ -ocut 3 -odix _%%a -obil

    lasgrid -i TEMP_FILES\12_chm\res_%CHM_RESOLUTION%\h_%%a\*.bil ^
          -merged ^
          -step %CHM_RESOLUTION% ^
          -epsg %EPSG% ^
          -ll %GRID_ORIGIN% ^
          -odir TEMP_FILES\12_chm\res_%CHM_RESOLUTION%\merged\ -obil
  )

lasgrid -i TEMP_FILES\12_chm\res_%CHM_RESOLUTION%\merged\*.bil ^
        -merged ^
        -highest ^
        -step %CHM_RESOLUTION% ^
        -epsg %EPSG% ^
        -ll %GRID_ORIGIN% ^
        -odir OUTPUT_FILES\CHM\ -obil -ocut 3 -odix _chm_%CHM_RESOLUTION%m