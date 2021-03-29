mkdir .\OUTPUT_FILES\RAS

:: all return point density
lasgrid -i TEMP_FILES\11_no_buffer\*.laz ^
            -keep_class %CLASS_GROUND% ^
            -merged ^
            -step %RAS_RESOLUTION% ^
            -epsg %EPSG% ^
            -point_density ^
            -ll %GRID_ORIGIN% ^
            -o OUTPUT_FILES\RAS\%DATE%_ground_point_density_r%RAS_RESOLUTION%m.bil -obil