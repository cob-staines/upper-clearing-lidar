:: las_04_canopy_raster.bat
:: dependencies
    :: NOISE_HEIGHT_THRESHOLD_LOW
    :: NOISE_HEIGHT_THRESHOLD_HIGH
    :: CLASS_NOISE
    :: VEGETATION_HEIGHT_THRESHOLD_LOW
    :: CLASS_VEGETATION
    :: NUM_CORES
    :: RESOLUTION_CANOPY

:: make output directories
mkdir .\TEMP_FILES\07_height
mkdir .\TEMP_FILES\08_vegetation
mkdir .\TEMP_FILES\09_chm\res_%CHM_RESOLUTION%\merged
mkdir .\OUTPUT_FILES\CHM
mkdir .\OUTPUT_FILES\CANOPY_RASTER

:: calculate point height from ground point TIN. Classify points below HEIGHT_THRESHOLD_LOW and above HEIGHT_THRESHOLD_HIGH as 7 (noise)
lasheight -i TEMP_FILES\06_ground\*.laz ^
    -replace_z ^
    -cores %NUM_CORES% ^
    -odir TEMP_FILES\07_height\ -olaz -ocut 3 -odix _07

las2las -i TEMP_FILES\07_height\*.laz ^
    -classify_z_below_as %NOISE_HEIGHT_THRESHOLD_LOW% %CLASS_NOISE% ^
    -classify_z_above_as %VEGETATION_HEIGHT_THRESHOLD_LOW% %CLASS_VEGETATION% ^
    -classify_z_above_as %NOISE_HEIGHT_THRESHOLD_HIGH% %CLASS_NOISE% ^
    -cores %NUM_CORES% ^
    -odir TEMP_FILES\08_vegetation\ -olaz -ocut 3 -odix _08

for %%a in (%CHM_LAYER_LIST%) do (

    mkdir .\TEMP_FILES\09_chm\res_%CHM_RESOLUTION%\h_%%a

    blast2dem -i TEMP_FILES\08_vegetation\*.laz ^
          -keep_class %CLASS_VEGETATION% ^
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
lascanopy -i TEMP_FILES\07_height\*.laz ^
		-merged ^
    -use_tile_bb ^
    -keep_class %CLASS_VEGETATION% ^
		-step %CHM_RESOLUTION% ^
    -epsg %EPSG% ^
		-d 0.5 2 4 10 50 -dns -cov ^
		-odir OUTPUT_FILES\CANOPY_RASTER\ -obil -ocut 3