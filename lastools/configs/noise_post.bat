SET NOISE_CANOPY_ISOLATION=10
SET NOISE_CANOPY_STEP_XY=.5
SET NOISE_CANOPY_STEP_Z=.1

lasnoise -i OUTPUT_FILES\LAS\19_149_all_200311_628000_5646525_vegetation.las ^
          -isolated %NOISE_ISOLATION% ^
          -step_xy %NOISE_STEP_XY% ^
          -step_z %NOISE_STEP_Z% ^
          -classify_as %CLASS_NOISE% ^
          -o OUTPUT_FILES\LAS\19_149_all_200311_628000_5646525_vegetation_iso_%NOISE_ISOLATION%_xy_%NOISE_STEP_XY%_z_%NOISE_STEP_Z%.las