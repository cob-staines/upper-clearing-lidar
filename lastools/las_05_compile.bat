:: las_03_remove_buffer.bat
:: dependencies
     :: NUM_CORES
     :: CLASS_VEGETATION

mkdir .\TEMP_FILES\11_no_buffer
mkdir .\OUTPUT_FILES\LAS

:: remove and merge for
:: vegetation
lastile -i TEMP_FILES\07_vegetation\*.laz ^
          -keep_first ^
          -remove_buffer ^
          -cores %NUM_CORES% ^
          -odir TEMP_FILES\11_no_buffer\ -olaz -ocut 3 -odix _vegetation

las2las -i TEMP_FILES\11_no_buffer\*_vegetation.laz ^
          -merged ^
          -odir OUTPUT_FILES\LAS\ -olas