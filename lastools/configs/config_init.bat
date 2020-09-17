:: config init

:: universal
:: folder containing batch files
SET DIR_BAT=C:\Users\Cob\index\educational\usask\research\masters\repos\upper-clearing-lidar\lastools

:: call las_proc
SET DATE=19_045
SET PRODUCT_ID=%DATE%_las_proc
call %DIR_BAT%\configs\las_proc.bat

SET DATE=19_050
SET PRODUCT_ID=%DATE%_las_proc
call %DIR_BAT%\configs\las_proc.bat

SET DATE=19_052
SET PRODUCT_ID=%DATE%_las_proc
call %DIR_BAT%\configs\las_proc.bat

SET DATE=19_107
SET PRODUCT_ID=%DATE%_las_proc
call %DIR_BAT%\configs\las_proc.bat

SET DATE=19_123
SET PRODUCT_ID=%DATE%_las_proc
call %DIR_BAT%\configs\las_proc.bat

SET DATE=19_149
SET PRODUCT_ID=%DATE%_las_proc
call %DIR_BAT%\configs\las_proc.bat
