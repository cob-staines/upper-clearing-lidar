SET DIR_BAT=C:\Users\Cob\index\educational\usask\research\masters\repos\upper-clearing-lidar\lastools
SET DIR_LASTOOLS=C:\Users\Cob\index\educational\usask\research\masters\code_lib\lastools\LAStools\bin;
set PATH=%PATH%;%DIR_LASTOOLS%


SET DATE=19_045
SET DIR_WORKING=C:\Users\Cob\index\educational\usask\research\masters\data\LiDAR\%DATE%
pushd %DIR_WORKING%
las2las -i 19_045_all_WGS84_utm11N_raw.las ^
	-drop_gps_time_between  414067 414069 ^
	-drop_gps_time_between  415602 415604 ^
	-o 19_045_all_WGS84_utm11N.las

SET DATE=19_050
SET DIR_WORKING=C:\Users\Cob\index\educational\usask\research\masters\data\LiDAR\%DATE%
pushd %DIR_WORKING%
las2las -i 19_050_all_WGS84_utm11N_raw.las ^
	-drop_gps_time_between  240737 240739 ^
	-drop_gps_time_between  241928 241930 ^
	-o 19_050_all_WGS84_utm11N.las

SET DATE=19_052
SET DIR_WORKING=C:\Users\Cob\index\educational\usask\research\masters\data\LiDAR\%DATE%
pushd %DIR_WORKING%
las2las -i 19_052_all_WGS84_utm11N_raw.las ^
	-drop_gps_time_between  411253 411255 ^
	-drop_gps_time_between  412358 412360 ^
	-o 19_052_all_WGS84_utm11N.las

SET DATE=19_107
SET DIR_WORKING=C:\Users\Cob\index\educational\usask\research\masters\data\LiDAR\%DATE%
pushd %DIR_WORKING%
las2las -i 19_107_all_WGS84_utm11N_raw.las ^
	-drop_gps_time_between  321260 321262 ^
	-drop_gps_time_between  323567 323569 ^
	-o 19_107_all_WGS84_utm11N.las

SET DATE=19_123
SET DIR_WORKING=C:\Users\Cob\index\educational\usask\research\masters\data\LiDAR\%DATE%
pushd %DIR_WORKING%
las2las -i 19_123_all_WGS84_utm11N_raw.las ^
	-drop_gps_time_between 494877 494879 ^
	-drop_gps_time_between 495916 495918 ^
	-drop_gps_time_between 495984 495986 ^
	-o 19_123_all_WGS84_utm11N.las

SET DATE=19_149
SET DIR_WORKING=C:\Users\Cob\index\educational\usask\research\masters\data\LiDAR\%DATE%
pushd %DIR_WORKING%
las2las -i 19_149_all_WGS84_utm11N_raw.las ^
	-drop_gps_time_between 324515 324517 ^
	-drop_gps_time_between 325507 325509 ^
	-drop_gps_time_between 326761 326763 ^
	-o 19_149_all_WGS84_utm11N.las