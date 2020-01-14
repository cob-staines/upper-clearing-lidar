workingdir <- "C:/Users/Cob/index/educational/usask/research/masters/data/surveys/"

tasks <- c("19_045", "19_050", "19_052", "19_107", "19_123")

snow_file <- "depth_swe/marmot_snow_surveys_raw_19_045.csv"
gnss_file <- "gnss/19_045_ground_points_corrected.csv"
output <- "19_045_snow_survey_merged.csv"

snow <- read.csv(paste0(workingdir,snow_file), header=TRUE, skip=1, na.strings = c("NA",""))
gnss <- read.csv(paste0(workingdir,gnss_file), header=TRUE, na.strings = c("NA",""), sep=";")

data <- merge(snow, gnss, by.x="gps_point", by.y="Point.Id")

write.csv(data, paste0(workingdir,output), row.names=FALSE, na="")
