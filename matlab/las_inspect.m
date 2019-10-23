%config
filedir = 'C:\Users\Cob\index\educational\usask\research\masters\data\LiDAR\19_045\';
filename = '19_045_grid_60m_WGS84_utm11N_nocolor.las';
path = [filedir, filename];

%%
%read file
c = lasdata(path, 'load_all');
%%
c.scan_angle = get_scan_angle(c);
c.intensity = get_intensity(c);
c.classification = get_classification(c);
% why are attributes blank? intensity, scanning angle, gpstime

%%
%write file