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