tasks <- c("19_045", "19_050", "19_052", "19_107", "19_123")
doy <- c(45, 50, 52, 107, 123)

snow_data = data.frame()
for (ii in 1:5) {
  snow_file <- paste0("C:/Users/Cob/index/educational/usask/research/masters/data/surveys/depth_swe/marmot_snow_surveys_raw_", tasks[ii] , ".csv")
  temp = read.csv(snow_file, header=TRUE, na.strings = c("NA",""), skip=1, sep=",")
  temp$doy = doy[ii]
  snow_data = rbind(snow_data, temp)
}

depth_data <- snow_data[,c(1, 2, 3, 4, 5, 10)]

gnss_file <- "C:/Users/Cob/index/educational/usask/research/masters/data/surveys/all_ground_points_UTM11N_uid_flagged_cover.csv"
gnss <- read.csv(gnss_file, header=TRUE, na.strings = c("NA",""), sep=",")
gnss$Date.Time <- as.POSIXct(gnss$Date.Time, format="%d/%m/%Y %H:%M:%OS")
gnss$doy <- as.numeric(strftime(gnss$Date.Time, format = "%j"))

gnss_depth <- merge(gnss, depth_data, by.x=c("Point.Id", "doy"), by.y=c("gps_point", "doy"), all.x = TRUE)

output <- "C:/Users/Cob/index/educational/usask/research/masters/data/surveys/depth_swe/snow_survey_gnss_merged.csv"
write.csv(gnss_depth, output, row.names=FALSE, na="")