:: las_00_dir_setup.bat
:: dependencies
	:: DIR_LASTOOLS
    :: DIR_WORKING
    :: PRODUCT_ID
    

:: include LAStools in PATH to allow running script from here
set PATH=%PATH%;%DIR_LASTOOLS%

:: initial setup
pushd %DIR_WORKING%

:: make product folder
mkdir .\%PRODUCT_ID%
cd %PRODUCT_ID%