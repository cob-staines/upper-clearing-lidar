
:: ground point density (last returns)

lasgrid -i TEMP_FILES\11_no_buffer\*_vegetation.laz ^
            -keep_last ^
            -keep_class %CLASS_GROUND% ^
            -merged ^
            -step 0.1 ^
            -epsg %EPSG% ^
            -point_density ^
            -odir OUTPUT_FILES\DEM\ -obil -ocut 3 -odix _point_density_0.1m

:: vegetation point density (last returns)

lasgrid -i TEMP_FILES\11_no_buffer\*_vegetation.laz ^
            -keep_last ^
            -keep_class %CLASS_VEGETATION% ^
            -merged ^
            -step 0.1 ^
            -epsg %EPSG% ^
            -point_density ^
            -odir OUTPUT_FILES\DEM\ -obil -ocut 3 -odix _veg_point_density_0.1m