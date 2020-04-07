::mkdir .\OUTPUT_FILES\DEM\offset_opt\
::mkdir .\TEMP_FILES\06_ground\offset_opt\step_%GROUND_STEP%_offset_%GROUND_OFFSET%\
::
:::: ground classified tiles
::lasground -i TEMP_FILES\05_noise\*.laz ^
::      	-step %GROUND_STEP% ^
::      	-offset %GROUND_OFFSET% ^
::      	-ignore_class %CLASS_NOISE% ^
::      	-cores %NUM_CORES% ^
::    	  -odir TEMP_FILES\06_ground\offset_opt\step_%GROUND_STEP%_offset_%GROUND_OFFSET%\ -olaz -ocut 3 -odix _06

mkdir .\TEMP_FILES\10_dem\offset_opt_res_%DEM_RESOLUTION%\step_%GROUND_STEP%_offset_%GROUND_OFFSET%\

:: build dem
blast2dem -i TEMP_FILES\06_ground\offset_opt\step_%GROUND_STEP%_offset_%GROUND_OFFSET%\*.laz ^
        -keep_class %CLASS_GROUND% ^
        -keep_last ^
        -use_tile_bb ^
        -thin_with_grid %DEM_THIN_RESOLUTION% ^
        -step %DEM_RESOLUTION% ^
        -kill %DEM_MAX_TIN_EDGE% ^
        -cores %NUM_CORES% ^
        -ll %GRID_ORIGIN% ^
        -odir  TEMP_FILES\10_dem\offset_opt_res_%DEM_RESOLUTION%\step_%GROUND_STEP%_offset_%GROUND_OFFSET%\ -obil -ocut 3 -odix _10

lasgrid -i TEMP_FILES\10_dem\offset_opt_res_%DEM_RESOLUTION%\step_%GROUND_STEP%_offset_%GROUND_OFFSET%\*.bil ^
        -merged ^
        -step %DEM_RESOLUTION% ^
        -epsg %EPSG% ^
        -ll %GRID_ORIGIN% ^
        -odir OUTPUT_FILES\DEM\offset_opt\ -obil -ocut 3 -odix dem_%DEM_RESOLUTION%_step_%GROUND_STEP%_offset_%GROUND_OFFSET%