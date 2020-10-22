:: las_03_remove_buffer.bat
:: dependencies
     :: NUM_CORES
     :: CLASS_VEGETATION
mkdir .\TEMP_FILES\10_sorted
mkdir .\TEMP_FILES\11_no_buffer
mkdir .\OUTPUT_FILES\LAS

:: remove and merge for
:: vegetation
lassort -i TEMP_FILES\08_classified\*.laz ^
        -gps_time ^
        -return_number ^
        -cores %NUM_CORES% ^
        -odir TEMP_FILES\10_sorted\ -olaz -ocut 3 -odix _10

lastile -i TEMP_FILES\10_sorted\*.laz ^
          -remove_buffer ^
          -cores %NUM_CORES% ^
          -odir TEMP_FILES\11_no_buffer\ -olaz -ocut 3 -odix _11

las2las -i TEMP_FILES\11_no_buffer\*.laz ^
          -merged ^
          -odir OUTPUT_FILES\LAS\ -olas -ocut 18 -odix _classified_merged


mkdir .\TEMP_FILES\10_ground_thinned_sorted
mkdir .\TEMP_FILES\11_ground_thinned_no_buffer

lassort -i TEMP_FILES\06_ground_thinned\*.laz ^
        -gps_time ^
        -return_number ^
        -cores %NUM_CORES% ^
        -odir TEMP_FILES\10_ground_thinned_sorted\ -olaz -ocut 3 -odix _10
:: remove buffer
lastile -i TEMP_FILES\10_ground_thinned_sorted\*.laz ^
          -remove_buffer ^
          -cores %NUM_CORES% ^
          -odir TEMP_FILES\11_ground_thinned_no_buffer\ -olaz -ocut 3 -odix _11

:: merged output of all ground points (snow off only)
las2las -i TEMP_FILES\11_ground_thinned_no_buffer\*.laz ^
          -merged ^
          -odir OUTPUT_FILES\LAS\ -olas -ocut 18 -odix _ground_thinned_merged
