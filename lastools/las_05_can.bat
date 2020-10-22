:: las_04_canopy_raster.bat
:: dependencies
    :: CHM_RESOLUTION
    :: CHM_MAX_TIN_EDGE
    :: CLASS_VEGETATION
    :: NUM_CORES
    :: EPSG

:: make output directories
::mkdir .\TEMP_FILES\18_veg
mkdir .\TEMP_FILES\13_dsm_canopy\res_%CAN_RESOLUTION%\
mkdir .\OUTPUT_FILES\CAN


:: spike-free CHM following Khosravipour et al. 2016, 2015, 2014

::las2las -i TEMP_FILES\10_normalized\*.laz ^
::      -keep_class %CLASS_VEGETATION% ^
::      -cores %NUM_CORES% ^
::      -odir TEMP_FILES\18_veg\ -ocut 3 -odix _18 -olaz

las2dem -i TEMP_FILES\08_classified\*.laz ^
      -use_tile_bb ^
      -drop_class %CLASS_NOISE% ^
      -spike_free %CAN_MAX_TIN_EDGE% ^
      -step %CAN_RESOLUTION% ^
      -kill %CAN_MAX_TIN_EDGE% ^
      -ll %GRID_ORIGIN% ^
      -cores %NUM_CORES% ^
      -odir TEMP_FILES\13_dsm_canopy\res_%CAN_RESOLUTION%\ -ocut 3 -odix _spike_free_dsm_can_%CAN_RESOLUTION%m -obil


:: merge output raster
lasgrid -i TEMP_FILES\13_dsm_canopy\res_%CAN_RESOLUTION%\*.bil ^
      -merged ^
      -step %CAN_RESOLUTION% ^
      -epsg %EPSG% ^
      -ll %GRID_ORIGIN% ^
      -o OUTPUT_FILES\CAN\%DATE%_spike_free_dsm_can_r%CAN_RESOLUTION%m.bil -obil