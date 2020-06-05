# compiles all gnss data from all survey days

workingdir <- "C:/Users/Cob/index/educational/usask/research/masters/data/surveys/"
output <- "all_ground_points.csv"

tasks <- c("19_045", "19_050", "19_052", "19_107", "19_123", "19_149")
gnss <- setNames(data.frame(matrix(ncol = 7, nrow = 0)), c("Point.Id", "Point.Role", "WGS84.Latitude", "WGS84.Longitude", "WGS84.Ellip..Height", "Date.Time", "CQ.3D"))
for (ii in tasks){
  gnss_file <- paste0(workingdir,"gnss/",ii,"_ground_points_corrected.csv")
  gnss <- rbind(gnss,read.csv(gnss_file, header=TRUE, na.strings = c("NA",""), sep=";"))
}

write.csv(gnss, paste0(workingdir,output), row.names=FALSE, na="")
