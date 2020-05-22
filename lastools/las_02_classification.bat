:: las_02_classification.bat
:: dependencies
     :: CLASS_GROUND
     :: CLASS_VEGETATION
     :: CLASS_NOISE
     :: NUM_CORES
     :: NOISE_ISOLATION
     :: NOISE_STEP
     :: GROUND_STEP
     :: NOISE_HEIGHT_THRESHOLD_LOW
     :: NOISE_HEIGHT_THRESHOLD_HIGH
     :: VEGETATION_HEIGHT_THRESHOLD_LOW

:: make output directories
mkdir .\TEMP_FILES\05_noise_ground
mkdir .\TEMP_FILES\06_ground
mkdir .\TEMP_FILES\07_vegetation
mkdir .\TEMP_FILES\08_classified
mkdir .\TEMP_FILES\09_normalized

:: identify noise for ground analysis
lasnoise -i TEMP_FILES\04_duplicate\*.laz ^
          -isolated %NOISE_GROUND_ISOLATION% ^
          -step_xy %NOISE_GROUND_STEP% ^
          -step_z %NOISE_GROUND_STEP% ^
          -classify_as %CLASS_NOISE_GROUND% ^
          -cores %NUM_CORES% ^
          -odir TEMP_FILES\05_noise_ground\ -olaz -ocut 3 -odix _05

:: ground classify tiles
lasground -i TEMP_FILES\05_noise_ground\*.laz ^
          -step %GROUND_STEP% ^
          -offset %GROUND_OFFSET% ^
          -ignore_class %CLASS_NOISE_GROUND% ^
          -cores %NUM_CORES% ^
          -odir TEMP_FILES\06_ground\ -olaz -ocut 3 -odix _06

:: calculate point height from ground point TIN.
:: Classify vegetation points between VEGETATION_HEIGHT_THRESHOLD_LOW and NOISE_HEIGHT_THRESHOLD_HIGH
:: below HEIGHT_THRESHOLD_LOW and above HEIGHT_THRESHOLD_HIGH as NOISE_CLASS
lasheight -i TEMP_FILES\06_ground\*.laz ^
          -ignore_class %CLASS_NOISE_GROUND% ^
          -classify_below  %NOISE_HEIGHT_THRESHOLD_LOW% %CLASS_NOISE_GROUND% ^
          -classify_between %VEGETATION_HEIGHT_THRESHOLD_LOW% %NOISE_HEIGHT_THRESHOLD_HIGH% %CLASS_VEGETATION% ^
          -classify_above %NOISE_HEIGHT_THRESHOLD_HIGH% %CLASS_NOISE_CANOPY% ^
          -cores %NUM_CORES% ^
          -odir TEMP_FILES\07_vegetation\ -olaz -ocut 3 -odix _07

:: canopy noise
lasnoise -i TEMP_FILES\07_vegetation\*.laz ^
          -ignore_class %CLASS_GROUND% ^
          -isolated %NOISE_CANOPY_ISOLATION% ^
          -step_xy %NOISE_CANOPY_STEP_XY% ^
          -step_z %NOISE_CANOPY_STEP_Z% ^
          -classify_as %CLASS_NOISE_CANOPY% ^
          -odir TEMP_FILES\08_classified\ -olaz -ocut 3 -odix _08

:: recalculate height and replace z to normalize by ground surface (required for CHM)
lasheight -i TEMP_FILES\08_classified\*.laz ^
          -ignore_class %CLASS_NOISE_GROUND% %CLASS_NOISE_CANOPY% ^
          -replace_z ^
          -cores %NUM_CORES% ^
          -odir TEMP_FILES\09_normalized\ -olaz -ocut 3 -odix _09

