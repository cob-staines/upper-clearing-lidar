call %DIR_BAT%\las_00_dir_setup.bat

:: __________ PROTOCOL__________

IF %BOOL_QC%==1 call %DIR_BAT%\las_01_quality_control.bat
     :: dependencies
          :: DIR_WORKING
          :: DIR_LASTOOLS
          :: DIR_SITE_LIBRARY
          :: PRODUCT_ID
          :: FILE_IN
          :: ORIGINAL_SCALE_FACTOR
          :: NUM_CORES
          :: TILE_SIZE
          :: TILE_BUFFER

IF %BOOL_CLASSIFY%==1 call %DIR_BAT%\las_02_classification.bat
     :: dependencies
          :: CLASS_GROUND
          :: CLASS_NOISE
          :: NUM_CORES
          :: NOISE_ISOLATION
          :: NOISE_STEP
          :: GROUND_STEP
          :: HEIGHT_THRESHOLD_LOW
          :: HEIGHT_THRESHOLD_HIGH

IF %BOOL_DEM%==1 call %DIR_BAT%\las_03_dem.bat
     :: dependencies
          :: CLASS_GROUND
          :: RESOLUTION_THIN
          :: RESOLUTION_DEM
          :: MAX_TIN_EDGE
          :: NUM_CORES
          :: EPSG

IF %BOOL_CHM%==1 call %DIR_BAT%\las_04_canopy.bat
:: dependencies
    :: NOISE_HEIGHT_THRESHOLD_LOW
    :: NOISE_HEIGHT_THRESHOLD_HIGH
    :: CLASS_NOISE
    :: VEGETATION_HEIGHT_THRESHOLD_LOW
    :: CLASS_VEGETATION
    :: NUM_CORES
    :: RESOLUTION_CANOPY
IF %BOOL_COMPILE%==1 call %DIR_BAT%\las_05_compile.bat
:: dependencies
    :: NUM_CORES

IF %RM_TEMP%==NOISE (
  rmdir /s /q .\TEMP_FILES\00_quality\
  rmdir /s /q. \TEMP_FILES\01_precision\
  rmdir /s /q .\TEMP_FILES\02_clip\
  rmdir /s /q .\TEMP_FILES\03_tile\
  rmdir /s /q .\TEMP_FILES\04_duplicate\
  rmdir /s /q .\TEMP_FILES\05_noise\
)
  :: __________ MANUAL OUTPUTS __________

