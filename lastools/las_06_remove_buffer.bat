mkdir .\TEMP_FILES\11_no_buffer

lastile -i TEMP_FILES\05_ground\*.laz ^
          -remove_buffer ^
          -cores %NUM_CORES% ^
          -odir TEMP_FILES\11_no_buffer\ -olaz -ocut 3 -odix _11