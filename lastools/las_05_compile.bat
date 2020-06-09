:: las_03_remove_buffer.bat
:: dependencies
     :: NUM_CORES
     :: CLASS_VEGETATION

mkdir .\TEMP_FILES\11_no_buffer
mkdir .\OUTPUT_FILES\LAS

:: remove and merge for
:: vegetation
lastile -i TEMP_FILES\08_classified\*.laz ^
          -remove_buffer ^
          -cores %NUM_CORES% ^
          -odir TEMP_FILES\11_no_buffer\ -olaz -ocut 3 -odix _11

las2las -i TEMP_FILES\11_no_buffer\*.laz ^
          -merged ^
          -odir OUTPUT_FILES\LAS\ -olas -ocut 18 -odix _clean
