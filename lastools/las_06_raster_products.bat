mkdir .\OUTPUT_FILES\RAS

:: all return point density
lasgrid -i TEMP_FILES\11_no_buffer\*.laz ^
            -drop_class %CLASS_NOISE_GROUND% %CLASS_NOISE_CANOPY% ^
            -keep_scan_angle -%RAS_MAX_ANGLE% %RAS_MAX_ANGLE% ^
            -merged ^
            -step %RAS_RESOLUTION% ^
            -epsg %EPSG% ^
            -point_density ^
            -ll %GRID_ORIGIN% ^
            -odir OUTPUT_FILES\RAS\ -obil -ocut 3 -odix _all_point_density_a%RAS_MAX_ANGLE%_r%RAS_RESOLUTION%m

:: first ground point density
lasgrid -i TEMP_FILES\11_no_buffer\*.laz ^
            -keep_class %CLASS_GROUND% ^
            -keep_first ^
            -keep_scan_angle -%RAS_MAX_ANGLE% %RAS_MAX_ANGLE% ^
            -merged ^
            -step %RAS_RESOLUTION% ^
            -epsg %EPSG% ^
            -point_density ^
            -ll %GRID_ORIGIN% ^
            -odir OUTPUT_FILES\RAS\ -obil -ocut 3 -odix _first_ground_point_density_a%RAS_MAX_ANGLE%_r%RAS_RESOLUTION%m

:: last ground point density
lasgrid -i TEMP_FILES\11_no_buffer\*.laz ^
            -keep_class %CLASS_GROUND% ^
            -keep_last ^
            -keep_scan_angle -%RAS_MAX_ANGLE% %RAS_MAX_ANGLE% ^
            -merged ^
            -step %RAS_RESOLUTION% ^
            -epsg %EPSG% ^
            -point_density ^
            -ll %GRID_ORIGIN% ^
            -odir OUTPUT_FILES\RAS\ -obil -ocut 3 -odix _last_ground_point_density_a%RAS_MAX_ANGLE%_r%RAS_RESOLUTION%m

:: first veg point density
lasgrid -i TEMP_FILES\11_no_buffer\*.laz ^
            -keep_class %CLASS_VEGETATION% ^
            -keep_first ^
            -keep_scan_angle -%RAS_MAX_ANGLE% %RAS_MAX_ANGLE% ^
            -merged ^
            -step %RAS_RESOLUTION% ^
            -epsg %EPSG% ^
            -point_density ^
            -ll %GRID_ORIGIN% ^
            -odir OUTPUT_FILES\RAS\ -obil -ocut 3 -odix _first_veg_point_density_a%RAS_MAX_ANGLE%_r%RAS_RESOLUTION%m

:: last veg point density
lasgrid -i TEMP_FILES\11_no_buffer\*.laz ^
            -keep_class %CLASS_VEGETATION% ^
            -keep_last ^
            -keep_scan_angle -%RAS_MAX_ANGLE% %RAS_MAX_ANGLE% ^
            -merged ^
            -step %RAS_RESOLUTION% ^
            -epsg %EPSG% ^
            -point_density ^
            -ll %GRID_ORIGIN% ^
            -odir OUTPUT_FILES\RAS\ -obil -ocut 3 -odix _last_veg_point_density_a%RAS_MAX_ANGLE%_r%RAS_RESOLUTION%m
