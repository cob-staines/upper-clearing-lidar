mkdir .\OUTPUT_FILES\RAS


:: ground point density (last returns)

lasgrid -i TEMP_FILES\11_no_buffer\*_vegetation.laz ^
            -keep_last ^
            -keep_class %CLASS_GROUND% ^
            -merged ^
            -step %RAS_RESOLUTION% ^
            -epsg %EPSG% ^
            -point_density ^
            -odir OUTPUT_FILES\RAS\ -obil -ocut 3 -odix _point_density_%RAS_RESOLUTION%m

:: vegetation point density (first returns)

lasgrid -i TEMP_FILES\11_no_buffer\*_vegetation.laz ^
            -keep_first ^
            -keep_class %CLASS_VEGETATION% ^
            -merged ^
            -step %RAS_RESOLUTION% ^
            -epsg %EPSG% ^
            -point_density ^
            -odir OUTPUT_FILES\RAS\ -obil -ocut 3 -odix _veg_point_density_%RAS_RESOLUTION%m