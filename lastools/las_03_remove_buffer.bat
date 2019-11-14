:: las_03_remove_buffer.bat
:: dependencies
     :: NUM_CORES

cd TEMP_FILES
mkdir .\09_no_buffer
cd ..

:: remove buffer
lastile -i TEMP_FILES\08_vegetation\*.laz ^
          -remove_buffer ^
          -cores %NUM_CORES% ^
          -odir TEMP_FILES\09_no_buffer\ -olaz -ocut 3 -odix _09