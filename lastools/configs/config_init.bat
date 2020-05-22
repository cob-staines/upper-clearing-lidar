:: config init

:: universal
:: folder containing batch files
SET DIR_BAT=C:\Users\Cob\index\educational\usask\research\masters\repos\upper-clearing-lidar\lastools

:: snow_on
SET DATE=19_045
SET PRODUCT_ID=%DATE%_snow_on
call %DIR_BAT%\configs\snow_on_proc.bat

SET DATE=19_050
SET PRODUCT_ID=%DATE%_snow_on
call %DIR_BAT%\configs\snow_on_proc.bat

SET DATE=19_052
SET PRODUCT_ID=%DATE%_snow_on
call %DIR_BAT%\configs\snow_on_proc.bat

SET DATE=19_107
SET PRODUCT_ID=%DATE%_snow_on
call %DIR_BAT%\configs\snow_on_proc.bat

SET DATE=19_123
SET PRODUCT_ID=%DATE%_snow_on
call %DIR_BAT%\configs\snow_on_proc.bat

:: snow_off
SET DATE=19_149
SET PRODUCT_ID=%DATE%_snow_off
call %DIR_BAT%\configs\snow_off_proc.bat
