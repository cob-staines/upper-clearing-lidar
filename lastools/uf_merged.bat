mkdir .\OUTPUT_FILES\LAS

SET PLOT_MASK=C:\Users\Cob\index\educational\usask\research\masters\data\LiDAR\site_library\sub_plot_library\forest_upper.shp


:: clip to uf and merge
lasclip -i TEMP_FILES\11_no_buffer\*.laz ^
		-merged ^
        -poly %PLOT_MASK% ^
        -o OUTPUT_FILES\LAS\%DATE%_UF.las