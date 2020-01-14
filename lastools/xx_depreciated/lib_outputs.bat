:: output ground points
lasmerge -i TEMP_FILES\08_no_buffer\*.laz ^
          -keep_class 2 ^
          -o OUTPUT_FILES\%PRODUCT_ID%_ground-points.laz

:: output high vegetation
lasmerge -i TEMP_FILES\08_no_buffer\*.laz ^
          -keep_class 5 ^
          -o OUTPUT_FILES\%PRODUCT_ID%_vegetation-points.laz

:: ladder_clearing
:: clip to upper clearing poly, filter by 1st return, filter to +/- 5 deg
lasclip -i OUTPUT_FILES\%PRODUCT_ID%_ground-points.laz ^
          -poly C:\Users\Cob\index\educational\usask\research\masters\data\LiDAR\site_library\upper_clearing_poly.shp ^
          -keep_single ^
          -keep_scan_angle -5 5 ^
          -o OUTPUT_FILES\%PRODUCT_ID%_clearing_ground-points_single-return_5deg.las