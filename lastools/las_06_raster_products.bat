mkdir .\OUTPUT_FILES\RAS

:: all return point density
lasgrid -i TEMP_FILES\11_no_buffer\*.laz ^
            -drop_class %CLASS_NOISE_GROUND% %CLASS_NOISE_CANOPY% ^
            -merged ^
            -step %RAS_RESOLUTION% ^
            -epsg %EPSG% ^
            -point_density ^
            -ll %GRID_ORIGIN% ^
            -odir OUTPUT_FILES\RAS\ -obil -ocut 3 -odix _all_point_density_%RAS_RESOLUTION%m

:: 1st return point density
lasgrid -i TEMP_FILES\11_no_buffer\*.laz ^
            -keep_first ^
            -drop_class %CLASS_NOISE_GROUND% %CLASS_NOISE_CANOPY% ^
            -merged ^
            -step %RAS_RESOLUTION% ^
            -epsg %EPSG% ^
            -point_density ^
            -ll %GRID_ORIGIN% ^
            -odir OUTPUT_FILES\RAS\ -obil -ocut 3 -odix _1st_return_point_density_%RAS_RESOLUTION%m

:: last return point density
lasgrid -i TEMP_FILES\11_no_buffer\*.laz ^
            -keep_last ^
            -drop_class %CLASS_NOISE_GROUND% %CLASS_NOISE_CANOPY% ^
            -merged ^
            -step %RAS_RESOLUTION% ^
            -epsg %EPSG% ^
            -point_density ^
            -ll %GRID_ORIGIN% ^
            -odir OUTPUT_FILES\RAS\ -obil -ocut 3 -odix _last_return_point_density_%RAS_RESOLUTION%m

:: single return point density
lasgrid -i TEMP_FILES\11_no_buffer\*.laz ^
            -keep_single ^
            -drop_class %CLASS_NOISE_GROUND% %CLASS_NOISE_CANOPY% ^
            -merged ^
            -step %RAS_RESOLUTION% ^
            -epsg %EPSG% ^
            -point_density ^
            -ll %GRID_ORIGIN% ^
            -odir OUTPUT_FILES\RAS\ -obil -ocut 3 -odix _single_return_point_density_%RAS_RESOLUTION%m

:: ground point density
lasgrid -i TEMP_FILES\11_no_buffer\*.laz ^
            -keep_class %CLASS_GROUND% ^
            -merged ^
            -step %RAS_RESOLUTION% ^
            -epsg %EPSG% ^
            -point_density ^
            -ll %GRID_ORIGIN% ^
            -odir OUTPUT_FILES\RAS\ -obil -ocut 3 -odix _ground_point_density_%RAS_RESOLUTION%m

:: vegetation point density
lasgrid -i TEMP_FILES\11_no_buffer\*.laz ^
            -keep_class %CLASS_VEGETATION% ^
            -merged ^
            -step %RAS_RESOLUTION% ^
            -epsg %EPSG% ^
            -point_density ^
            -ll %GRID_ORIGIN% ^
            -odir OUTPUT_FILES\RAS\ -obil -ocut 3 -odix _veg_point_density_%RAS_RESOLUTION%m

:: first ground point density
lasgrid -i TEMP_FILES\11_no_buffer\*.laz ^
            -keep_class %CLASS_GROUND% ^
            -keep_first ^
            -keep_scan_angle -30 30 ^
            -merged ^
            -step %RAS_RESOLUTION% ^
            -epsg %EPSG% ^
            -point_density ^
            -ll %GRID_ORIGIN% ^
            -odir OUTPUT_FILES\RAS\ -obil -ocut 3 -odix _first_ground_point_density_d30_%RAS_RESOLUTION%m

:: last ground point density
lasgrid -i TEMP_FILES\11_no_buffer\*.laz ^
            -keep_class %CLASS_GROUND% ^
            -keep_last ^
            -keep_scan_angle -30 30 ^
            -merged ^
            -step %RAS_RESOLUTION% ^
            -epsg %EPSG% ^
            -point_density ^
            -ll %GRID_ORIGIN% ^
            -odir OUTPUT_FILES\RAS\ -obil -ocut 3 -odix _last_ground_point_density_d30_%RAS_RESOLUTION%m

:: first veg point density
lasgrid -i TEMP_FILES\11_no_buffer\*.laz ^
            -keep_class %CLASS_VEGETATION% ^
            -keep_first ^
            -keep_scan_angle -30 30 ^
            -merged ^
            -step %RAS_RESOLUTION% ^
            -epsg %EPSG% ^
            -point_density ^
            -ll %GRID_ORIGIN% ^
            -odir OUTPUT_FILES\RAS\ -obil -ocut 3 -odix _first_veg_point_density_d30_%RAS_RESOLUTION%m

:: last veg point density
lasgrid -i TEMP_FILES\11_no_buffer\*.laz ^
            -keep_class %CLASS_VEGETATION% ^
            -keep_last ^
            -keep_scan_angle -30 30 ^
            -merged ^
            -step %RAS_RESOLUTION% ^
            -epsg %EPSG% ^
            -point_density ^
            -ll %GRID_ORIGIN% ^
            -odir OUTPUT_FILES\RAS\ -obil -ocut 3 -odix _last_veg_point_density_d30_%RAS_RESOLUTION%m

:: first ground point density
lasgrid -i TEMP_FILES\11_no_buffer\*.laz ^
            -keep_class %CLASS_GROUND% ^
            -keep_first ^
            -keep_scan_angle -15 15 ^
            -merged ^
            -step %RAS_RESOLUTION% ^
            -epsg %EPSG% ^
            -point_density ^
            -ll %GRID_ORIGIN% ^
            -odir OUTPUT_FILES\RAS\ -obil -ocut 3 -odix _first_ground_point_density_d15_%RAS_RESOLUTION%m

:: last ground point density
lasgrid -i TEMP_FILES\11_no_buffer\*.laz ^
            -keep_class %CLASS_GROUND% ^
            -keep_last ^
            -keep_scan_angle -15 15 ^
            -merged ^
            -step %RAS_RESOLUTION% ^
            -epsg %EPSG% ^
            -point_density ^
            -ll %GRID_ORIGIN% ^
            -odir OUTPUT_FILES\RAS\ -obil -ocut 3 -odix _last_ground_point_density_d15_%RAS_RESOLUTION%m

:: first veg point density
lasgrid -i TEMP_FILES\11_no_buffer\*.laz ^
            -keep_class %CLASS_VEGETATION% ^
            -keep_first ^
            -keep_scan_angle -15 15 ^
            -merged ^
            -step %RAS_RESOLUTION% ^
            -epsg %EPSG% ^
            -point_density ^
            -ll %GRID_ORIGIN% ^
            -odir OUTPUT_FILES\RAS\ -obil -ocut 3 -odix _first_veg_point_density_d15_%RAS_RESOLUTION%m

:: last veg point density
lasgrid -i TEMP_FILES\11_no_buffer\*.laz ^
            -keep_class %CLASS_VEGETATION% ^
            -keep_last ^
            -keep_scan_angle -15 15 ^
            -merged ^
            -step %RAS_RESOLUTION% ^
            -epsg %EPSG% ^
            -point_density ^
            -ll %GRID_ORIGIN% ^
            -odir OUTPUT_FILES\RAS\ -obil -ocut 3 -odix _last_veg_point_density_d15_%RAS_RESOLUTION%m

:: analysis of height-normalized point cloud

:: mean height of 1st returns
:: mean height of last returns
:: standard deviation of all returns