:: initial setup
pushd %DIR_WORKING%

:: make product folder
mkdir .\%PRODUCT_ID%
cd %PRODUCT_ID%

:: make temp folder
mkdir .\TEMP_FILES
mkdir .\OUTPUT_FILES
cd TEMP_FILES

:: make output folders
mkdir .\01_precision
mkdir .\02_clip
mkdir .\03_tile
mkdir .\04_duplicate
mkdir .\05_noise
mkdir .\06_ground
mkdir .\07_vegetation
mkdir .\08_no_buffer