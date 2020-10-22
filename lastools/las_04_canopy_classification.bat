mkdir .\TEMP_FILES\07_vegetation
mkdir .\TEMP_FILES\08_classified
:: mkdir .\TEMP_FILES\09_sorted
:: mkdir .\TEMP_FILES\10_normalized

:: calculate point height from ground point TIN.
:: Classify vegetation points between VEGETATION_HEIGHT_THRESHOLD_LOW and NOISE_HEIGHT_THRESHOLD_HIGH
:: below HEIGHT_THRESHOLD_LOW and above HEIGHT_THRESHOLD_HIGH as NOISE_CLASS
lasheight -i TEMP_FILES\05_ground\*.laz ^
          -classify_below  %NOISE_HEIGHT_THRESHOLD_LOW% %CLASS_NOISE% ^
          -classify_between %VEGETATION_HEIGHT_THRESHOLD_LOW% %NOISE_HEIGHT_THRESHOLD_HIGH% %CLASS_VEGETATION% ^
          -classify_above %NOISE_HEIGHT_THRESHOLD_HIGH% %CLASS_NOISE% ^
          -cores %NUM_CORES% ^
          -odir TEMP_FILES\07_vegetation\ -olaz -ocut 3 -odix _07

:: classify canopy noise
lasnoise -i TEMP_FILES\07_vegetation\*.laz ^
          -isolated %NOISE_CANOPY_ISOLATION% ^
          -step_xy %NOISE_CANOPY_STEP_XY% ^
          -step_z %NOISE_CANOPY_STEP_Z% ^
          -classify_as %CLASS_NOISE% ^
          -odir TEMP_FILES\08_classified\ -olaz -ocut 3 -odix _08


:::: sort to speed up spike-free chm
:: lassort -i TEMP_FILES\08_classified\*.laz ^
::           -bucket 2 ^
::           -just_reorder ^
::           -cores %NUM_CORES% ^
::           -odir TEMP_FILES\09_sorted\ -olaz -ocut 3 -odix _09


:::: recalculate height and replace z to normalize by ground surface (required for CHM)
::lasheight -i TEMP_FILES\09_sorted\*.laz ^
::          -ignore_class %CLASS_NOISE% ^
::          -replace_z ^
::          -cores %NUM_CORES% ^
::          -odir TEMP_FILES\10_normalized\ -olaz -ocut 3 -odix _10
