mkdir .\OUTPUT_FILES\RAS

:: all return point density
lasgrid -i TEMP_FILES\11_no_buffer\*.laz ^
            -drop_class %CLASS_NOISE% ^
            -keep_scan_angle -%RAS_MAX_ANGLE% %RAS_MAX_ANGLE% ^
            -merged ^
            -step %RAS_RESOLUTION% ^
            -epsg %EPSG% ^
            -point_density ^
            -ll %GRID_ORIGIN% ^
            -o OUTPUT_FILES\RAS\%DATE%_all_point_density_a%RAS_MAX_ANGLE%_r%RAS_RESOLUTION%m.bil -obil

:: first point density
lasgrid -i TEMP_FILES\11_no_buffer\*.laz ^
            -drop_class %CLASS_NOISE% ^
            -keep_first ^
            -keep_scan_angle -%RAS_MAX_ANGLE% %RAS_MAX_ANGLE% ^
            -merged ^
            -step %RAS_RESOLUTION% ^
            -epsg %EPSG% ^
            -point_density ^
            -ll %GRID_ORIGIN% ^
            -o OUTPUT_FILES\RAS\%DATE%_first_point_density_a%RAS_MAX_ANGLE%_r%RAS_RESOLUTION%m.bil -obil

:: ground point density
lasgrid -i TEMP_FILES\11_no_buffer\*.laz ^
            -keep_class %CLASS_GROUND% ^
            -keep_scan_angle -%RAS_MAX_ANGLE% %RAS_MAX_ANGLE% ^
            -merged ^
            -step %RAS_RESOLUTION% ^
            -epsg %EPSG% ^
            -point_density ^
            -ll %GRID_ORIGIN% ^
            -o OUTPUT_FILES\RAS\%DATE%_ground_point_density_a%RAS_MAX_ANGLE%_r%RAS_RESOLUTION%m.bil -obil

:: first ground point density
::lasgrid -i TEMP_FILES\11_no_buffer\*.laz ^
::            -keep_class %CLASS_GROUND% ^
::            -keep_first ^
::            -keep_scan_angle -%RAS_MAX_ANGLE% %RAS_MAX_ANGLE% ^
::            -merged ^
::            -step %RAS_RESOLUTION% ^
::            -epsg %EPSG% ^
::            -point_density ^
::            -ll %GRID_ORIGIN% ^
::            -o OUTPUT_FILES\RAS\%DATE%_first_ground_point_density_a%RAS_MAX_ANGLE%_r%RAS_RESOLUTION%m.bil -obil
::
:::: last ground point density
::lasgrid -i TEMP_FILES\11_no_buffer\*.laz ^
::            -keep_class %CLASS_GROUND% ^
::            -keep_last ^
::            -keep_scan_angle -%RAS_MAX_ANGLE% %RAS_MAX_ANGLE% ^
::            -merged ^
::            -step %RAS_RESOLUTION% ^
::            -epsg %EPSG% ^
::            -point_density ^
::            -ll %GRID_ORIGIN% ^
::            -o OUTPUT_FILES\RAS\%DATE%_last_ground_point_density_a%RAS_MAX_ANGLE%_r%RAS_RESOLUTION%m.bil -obil
::
:::: first veg point density
::lasgrid -i TEMP_FILES\11_no_buffer\*.laz ^
::            -keep_class %CLASS_VEGETATION% ^
::            -keep_first ^
::            -keep_scan_angle -%RAS_MAX_ANGLE% %RAS_MAX_ANGLE% ^
::            -merged ^
::            -step %RAS_RESOLUTION% ^
::            -epsg %EPSG% ^
::            -point_density ^
::            -ll %GRID_ORIGIN% ^
::            -o OUTPUT_FILES\RAS\%DATE%_first_veg_point_density_a%RAS_MAX_ANGLE%_r%RAS_RESOLUTION%m.bil -obil
::
:::: last veg point density
::lasgrid -i TEMP_FILES\11_no_buffer\*.laz ^
::            -keep_class %CLASS_VEGETATION% ^
::            -keep_last ^
::            -keep_scan_angle -%RAS_MAX_ANGLE% %RAS_MAX_ANGLE% ^
::            -merged ^
::            -step %RAS_RESOLUTION% ^
::            -epsg %EPSG% ^
::            -point_density ^
::            -ll %GRID_ORIGIN% ^
::            -o OUTPUT_FILES\RAS\%DATE%_last_veg_point_density_a%RAS_MAX_ANGLE%_r%RAS_RESOLUTION%m.bil -obil
