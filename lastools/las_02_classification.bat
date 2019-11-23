:: las_02_classification.bat
:: dependencies
     :: CLASS_GROUND
     :: CLASS_NOISE
     :: NUM_CORES
     :: NOISE_ISOLATION
     :: NOISE_STEP
     :: GROUND_STEP
     :: HEIGHT_THRESHOLD_LOW
     :: HEIGHT_THRESHOLD_HIGH

cd TEMP_FILES
mkdir .\05_noise
mkdir .\06_ground
mkdir .\07_height
mkdir .\08_vegetation
cd ..

:: classify isolated points as class = 7
lasnoise -i TEMP_FILES\04_duplicate\*.laz ^
          -isolated %NOISE_ISOLATION% ^
          -step_xy %NOISE_STEP% ^
          -step_z %NOISE_STEP% ^
          -classify_as %CLASS_NOISE% ^
          -cores %NUM_CORES% ^
          -odir TEMP_FILES\05_noise\ -olaz -ocut 3 -odix _05

:: ground classify tiles
lasground -i TEMP_FILES\05_noise\*.laz ^
          -step %GROUND_STEP% ^
          -ignore_class %CLASS_NOISE% ^
          -cores %NUM_CORES% ^
          -odir TEMP_FILES\06_ground\ -olaz -ocut 3 -odix _06

:: calculate point height from ground point TIN. Classify points below HEIGHT_THRESHOLD_LOW and above HEIGHT_THRESHOLD_HIGH as 7 (noise)
lasheight -i TEMP_FILES\06_ground\*.laz ^
          -classify_below %HEIGHT_THRESHOLD_LOW% %CLASS_NOISE% -classify_above %HEIGHT_THRESHOLD_HIGH% %CLASS_NOISE% ^
          -cores %NUM_CORES% ^
          -odir TEMP_FILES\07_height\ -olaz -ocut 3 -odix _07

:: classify vegetation points as low vegetration, high vegetation (2m height threshold by default). Is this needed? I think not, no buildings!
lasclassify -i TEMP_FILES\07_height\*.laz ^
          -ignore_class %CLASS_GROUND% %CLASS_NOISE% ^
          -cores %NUM_CORES% ^
          -odir TEMP_FILES\08_vegetation\ -olaz -ocut 3 -odix _08
