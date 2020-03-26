library(dplyr)

# assigns codes to hemispherical photo lookup to allow for quick filtering later

wkdir <- "C:/Users/Cob/index/educational/usask/research/masters/data/"
hemi_in <- "hemispheres/hemi_lookup_coded.csv"
gps_in <- "surveys/all_ground_points_UTM11N.csv"
clean_out <- "hemispheres/hemi_lookup_cleaned.csv"

hemi <- read.csv(paste0(wkdir,hemi_in), header=TRUE, na.strings = c("NA",""), sep=",")
gps <- read.csv(paste0(wkdir,gps_in), header=TRUE, na.strings = c("NA",""), sep=",")

gps$Date.Time <- as.POSIXct(gps$Date.Time, format="%d/%m/%Y %H:%M:%OS")
gps$date <- as.Date(gps$Date.Time) - 1

date = unique(gps$date)
doy = c("19_045", "19_050", "19_052", "19_107", "19_123", "19_149")

lookup <- data.frame(date, doy)

gps <- merge(gps, lookup, by = "date", all.x = TRUE)

hemi <- merge(hemi, gps, by.x = c("gps_date", "gps_point"), by.y = c("doy", "Point.Id"), all.x = TRUE)

hemi$quality_code[hemi$quality_code < 9  & is.na(hemi$xcoordUTM1)] <- 8
hemi$quality_code[hemi$quality_code < 8   & hemi$Point.Rol == "GNSSCodeMeasuredRTK"] <- 7


clean <- hemi %>%
  select(folder, filename, quality_code, quality_note, height_m, WGS84.Lat, WGS84.Lon, WGS84.Ell, xcoordUTM1, ycoordUTM1
         , CQ.3D)

write.csv(clean, paste0(wkdir,clean_out), row.names = FALSE)