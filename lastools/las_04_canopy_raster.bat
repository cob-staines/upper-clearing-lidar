:: las_04_canopy_raster.bat
:: dependencies
    :: NUM_CORES
    :: RESOLUTION_CANOPY


cd TEMP_FILES
mkdir .\11_vegetation_normalized
cd ..

mkdir OUTPUT_FILES

cd OUTPUT_FILES
mkdir .\CANOPY_RASTER
cd ..

lasheight -i TEMP_FILES\09_no_buffer\*.laz ^
		-replace_z ^
		-cores %NUM_CORES% ^
		-odir  TEMP_FILES\11_vegetation_normalized -olaz -ocut 3 -odix _11

:: calculate raster canopy metrics
lascanopy -i TEMP_FILES\11_vegetation_normalized\*.laz ^
		-merged ^
		-step %RESOLUTION_CANOPY% ^
		-d 0.5 2 4 10 50 -dns -cov ^
		-odir OUTPUT_FILES\CANOPY_RASTER\ -obil -ocut 3