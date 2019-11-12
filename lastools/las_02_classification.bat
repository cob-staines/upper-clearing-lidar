:: las_02_classification.bat
:: dependencies
     :: NUM_CORES
     :: NOISE_ISOLATION
     :: NOISE_STEP
     :: GROUND_STEP

cd TEMP_FILES
mkdir .\05_noise
mkdir .\06_ground
mkdir .\07_vegetation
cd ..

:: classify isolated points as class = 7
lasnoise -i TEMP_FILES\04_duplicate\*.laz ^
          -isolated %NOISE_ISOLATION% ^
          -step_xy %NOISE_STEP% ^
          -step_z %NOISE_STEP% ^
          -classify_as 7 ^
          -cores %NUM_CORES% ^
          -odir TEMP_FILES\05_noise\ -olaz -ocut 3 -odix _05

:: ground classify tiles
lasground -i TEMP_FILES\05_noise\*.laz ^
          -step %GROUND_STEP% ^
          -ignore_class 7 ^
          -compute_height ^
          -cores %NUM_CORES% ^
          -odir TEMP_FILES\06_ground\ -olaz -ocut 3 -odix _06

lasclassify -i TEMP_FILES\06_ground\*.laz ^
          -ignore_class 2 7 ^
          -odir TEMP_FILES\07_vegetation\ -olaz -ocut 3 -odix _07
