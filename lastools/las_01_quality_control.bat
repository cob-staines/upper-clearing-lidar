:: las_01_quality_control.bat
:: dependencies
     :: FILE_IN
     :: PRODUCT_ID
     :: ORIGINAL_SCALE_FACTOR
     :: SITE_MASK
     :: NUM_CORES
     :: TILE_SIZE
     :: TILE_BUFFER

:: make output folders
mkdir .\TEMP_FILES\00_quality
mkdir .\TEMP_FILES\01_precision
mkdir .\TEMP_FILES\02_clip
mkdir .\TEMP_FILES\03_tile
mkdir .\TEMP_FILES\04_duplicate

:: ----------PROTOCOL----------


:: need to parse output to catch fails, warnings, etc.
lasinfo -i %FILE_IN% ^
        -compute_density ^
        -odir TEMP_FILES\00_quality -otxt

lasvalidate -i %FILE_IN% ^
            -no_CRS_fail ^
            -o TEMP_FILES\00_quality\validate.xml

:: should check each file prior to rescaling to verify actual precision
lasprecision -i %FILE_IN% ^
          -rescale %ORIGINAL_SCALE_FACTOR% %ORIGINAL_SCALE_FACTOR% %ORIGINAL_SCALE_FACTOR% ^
          -odir TEMP_FILES\01_precision\ -o %PRODUCT_ID%_01.laz

:: clip las by shpfile
lasclip -i TEMP_FILES\01_precision\%PRODUCT_ID%_01.laz ^
          -poly %SITE_MASK% ^
          -odir TEMP_FILES\02_clip\ -ocut 3 -odix _02 -olaz

:: tile las for memory management
lastile -i TEMP_FILES\02_clip\%PRODUCT_ID%_02.laz ^
          -set_classification 0 -set_user_data 0 ^
          -tile_size %TILE_SIZE% -buffer %TILE_BUFFER% ^
          -odir TEMP_FILES\03_tile\ -o %PRODUCT_ID%.laz

:: remove xyz duplicate points
lasduplicate -i TEMP_FILES\03_tile\*.laz ^
                -unique_xyz ^
                -cores %NUM_CORES% ^
                -odir TEMP_FILES\04_duplicate\ -olaz -odix _04
