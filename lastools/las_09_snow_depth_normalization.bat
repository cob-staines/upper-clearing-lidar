mkdir .\TEMP_FILES\14_ground_normalized

:: normalize snow-on ground points to snow-off TIN (for each snow-on point cloud)
lasheight -i TEMP_FILES\06_ground_thinned\*.laz ^
          -keep_class %CLASS_GROUND% ^
          -replace_z ^
          -ground_points %GROUND_POINTS_FILE% ^
        -cores %NUM_CORES% ^
        -odir TEMP_FILES\14_ground_normalized\ -olaz -ocut 3 -odix _07