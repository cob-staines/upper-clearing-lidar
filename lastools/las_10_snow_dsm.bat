mkdir .\TEMP_FILES\15_hs\res_%DSM_RESOLUTION%\
:: creat TIN and then rasterize snow depth surface (for each resolution, each snow-on point cloud)
blast2dem -i TEMP_FILES\14_ground_normalized\*.laz ^
        -keep_class %CLASS_GROUND% ^
        -use_tile_bb ^
        -step %DSM_RESOLUTION% ^
        -kill %DSM_MAX_TIN_EDGE% ^
        -float_precision %ORIGINAL_SCALE_FACTOR% ^
        -ll %GRID_ORIGIN% ^
        -cores %NUM_CORES% ^
        -odir  TEMP_FILES\15_hs\res_%DSM_RESOLUTION%\ -obil -ocut 3 -odix _12

