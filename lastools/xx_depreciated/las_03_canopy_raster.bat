:: las_03_canopy_raster.bat
:: dependencies
	:: CLASS_GROUND
    :: NUM_CORES
    :: RESOLUTION_DEM
    :: RESOLUTION_THIN
    :: MAX_TIN_EDGE


cd TEMP_FILES
mkdir .\11_vegetation_normalized
mkdir .\12_vegetation_normalized_merged
cd ..

mkdir OUTPUT_FILES

cd OUTPUT_FILES
mkdir .\CANOPY_RASTER
cd ..

lasheight -i TEMP_FILES\09_no_buffer\*.laz ^
		-replace_z ^
		-cores %NUM_CORES% ^
		-odir  TEMP_FILES\11_vegetation_normalized -olaz -ocut 3 -odix _11

:: output high vegetation
lasmerge -i TEMP_FILES\11_vegetation_normalized\*.laz ^
		-keep_class 5 ^
        -o TEMP_FILES\12_vegetation_normalized_merged\%PRODUCT_ID%_vegetation_normalized_12.laz

:: calculate raster canopy metrics
lascanopy -i TEMP_FILES\12_vegetation_normalized_merged\%PRODUCT_ID%_vegetation_normalized_12.laz ^
		-step .25 ^
		-cov -avg ^
		-odir OUTPUT_FILES\CANOPY_RASTER\ -obil -ocut 3 -odix canopy_cov