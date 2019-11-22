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
		-o TEMP_FILES\11_vegetation_normalized\%PRODUCT_ID%_vegetation_normalized_11.laz

:: calculate raster canopy metrics
lascanopy -i TEMP_FILES\11_vegetation_normalized_merged\%PRODUCT_ID%_vegetation_normalized_12.laz ^
		-merged ^
		-step %RESOLUTION_CANOPY% ^
		-dns ^
		-odir OUTPUT_FILES\CANOPY_RASTER\ -obil -ocut 3