infile <- "C:/Users/Cob/index/educational/usask/research/masters/data/surveys/all_ground_points_UTM11N.csv"
outfile <- "C:/Users/Cob/index/educational/usask/research/masters/data/surveys/all_ground_points_UTM11N_uid.csv"
data <- read.csv(infile, header=TRUE, na.strings = c("NA",""), sep=",")
data$uid = 1:(nrow(data))
write.csv(data, outfile, row.names=FALSE, na="")
