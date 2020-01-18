:: las_04_canopy_raster.bat
:: dependencies
    :: CHM_RESOLUTION
    :: CHM_MAX_TIN_EDGE
    :: CLASS_VEGETATION
    :: NUM_CORES
    :: EPSG

:: make output directories
mkdir .\TEMP_FILES\09_chm\res_%CHM_RESOLUTION%\merged
mkdir .\OUTPUT_FILES\CHM
mkdir .\OUTPUT_FILES\CANOPY_RASTER



for %%a in (%CHM_LAYER_LIST%) do (

    mkdir .\TEMP_FILES\09_chm\res_%CHM_RESOLUTION%\h_%%a

    blast2dem -i TEMP_FILES\08_normalized\*.laz ^
          -drop_class %CLASS_NOISE% ^
          -keep_first ^
          -drop_z_below %%a ^
          -use_tile_bb ^
          -step %CHM_RESOLUTION% ^
          -kill %CHM_MAX_TIN_EDGE% ^
          -cores %NUM_CORES% ^
          -odir TEMP_FILES\09_chm\res_%CHM_RESOLUTION%\h_%%a\ -ocut 3 -odix _%%a -obil

    lasgrid -i TEMP_FILES\09_chm\res_%CHM_RESOLUTION%\h_%%a\*.bil ^
          -merged ^
          -step %CHM_RESOLUTION% ^
          -epsg %EPSG% ^
          -odir TEMP_FILES\09_chm\res_%CHM_RESOLUTION%\merged\ -obil
  )

lasgrid -i TEMP_FILES\09_chm\res_%CHM_RESOLUTION%\merged\*.bil ^
        -merged ^
        -highest ^
        -step %CHM_RESOLUTION% ^
        -epsg %EPSG% ^
        -odir OUTPUT_FILES\CHM\ -obil -ocut 3 -odix pit_free_chm_%CHM_RESOLUTION%m

 

:: calculate raster canopy metrics
lascanopy -i TEMP_FILES\08_normalized\*.laz ^
		-merged ^
    -use_tile_bb ^
    -keep_class %CLASS_VEGETATION% ^
		-step %CHM_RESOLUTION% ^
    -epsg %EPSG% ^
		-d 0.5 2 4 10 50 -dns -cov ^
		-odir OUTPUT_FILES\CANOPY_RASTER\ -obil -ocut 3