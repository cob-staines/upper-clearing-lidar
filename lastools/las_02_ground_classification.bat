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
mkdir .\TEMP_FILES\05_ground
mkdir .\TEMP_FILES\06_ground_thinned


:: classify ground points
lasground_new -i TEMP_FILES\04_duplicate\*.laz ^
          -step %GROUND_STEP% ^
          -offset %GROUND_OFFSET% ^
          -spike %GROUND_SPIKE% ^
          -spike_down %GROUND_SPIKE% ^
          -ultra_fine ^
          -cores %NUM_CORES% ^
          -odir TEMP_FILES\05_ground\ -olaz -ocut 3 -odix _05

:: thin ground points to grid by percentile for blast2dem
lasthin -i TEMP_FILES\05_ground\*laz ^
         -keep_class %CLASS_GROUND% ^
         -step %GROUND_THIN_STEP% ^
         -percentile %GROUND_THIN_PERCENTILE% ^
         -cores %NUM_CORES% ^
         -odir TEMP_FILES\06_ground_thinned\ -olaz -ocut 3 -odix _06

