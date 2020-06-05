pnts_in <- "C:/Users/Cob/index/educational/usask/research/masters/data/surveys/all_ground_points_UTM11N_uid.csv"

filtered_out <- "C:/Users/Cob/index/educational/usask/research/masters/data/surveys/all_ground_points_UTM11N_uid_filtered.csv"
flagged_out <- "C:/Users/Cob/index/educational/usask/research/masters/data/surveys/all_ground_points_UTM11N_uid_flagged.csv"

#import all points
pnts = read.csv(pnts_in, header=TRUE, na.strings = c("NA",""), sep=",")

# filter points: drop phase-measured and hand-selected
filter = (pnts$Point.Rol == "GNSSPhaseMeasuredRTK") & (pnts$uid != 164) & (pnts$uid != 154) & (pnts$uid != 98) & (pnts$uid != 158) & (pnts$uid != 100) & (pnts$uid != 95)

# 0 is clean, 1 is flagged for qc
pnts_flagged = pnts
pnts_flagged$qc_flag = 0
pnts_flagged$qc_flag[!filter] = 1

pnts_filtered = pnts[filter,]

write.csv(pnts_flagged, flagged_out, row.names=FALSE, na="")
write.csv(pnts_filtered, filtered_out, row.names=FALSE, na="")

