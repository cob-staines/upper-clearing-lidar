SET PRODUCT_ID=%DATE_LIST%_all_200311
:: folder in which temp and output files will be saved to
SET DIR_WORKING=C:\Users\Cob\index\educational\usask\research\masters\data\LiDAR\%DATE_LIST%
SET FILE_IN=C:\Users\Cob\index\educational\usask\research\masters\data\LiDAR\%DATE_LIST%\%DATE_LIST%_all_WGS84_utm11N.las

call %DIR_BAT%\base_task_flow.bat