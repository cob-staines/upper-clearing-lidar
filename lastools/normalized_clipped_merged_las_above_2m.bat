mkdir .\OUTPUT_FILES\LAS
mkdir .\TEMP_FILES\15_normalized_no_buffer

SET PLOT_MASK=C:\Users\Cob\index\educational\usask\research\masters\data\LiDAR\site_library\sub_plot_library\forest_upper.shp


:: remove and merge for
:: vegetation
lastile -i TEMP_FILES\10_normalized\*.laz ^
          -remove_buffer ^
          -cores %NUM_CORES% ^
          -odir TEMP_FILES\15_normalized_no_buffer\ -olaz -ocut 3 -odix _15

:: clip to uf and merge
lasclip -i TEMP_FILES\15_normalized_no_buffer\*.laz ^
		-merged ^
		-keep_class %CLASS_VEGETATION% ^
		-drop_z_below 2 ^
        -poly %PLOT_MASK% ^
        -o OUTPUT_FILES\LAS\%DATE%_normalized_UF_above_2m.las