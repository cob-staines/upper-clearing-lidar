mkdir .\OUTPUT_FILES\TEMPLATES

:: all return point density
lasgrid -i TEMP_FILES\11_no_buffer\*.laz ^
            -drop_class %CLASS_NOISE% ^
            -merged ^
            -step %RAS_RESOLUTION% ^
            -epsg %EPSG% ^
            -point_density ^
            -ll %GRID_ORIGIN% ^
            -o OUTPUT_FILES\TEMPLATES\%DATE%_all_point_density_r%RAS_RESOLUTION%m.bil