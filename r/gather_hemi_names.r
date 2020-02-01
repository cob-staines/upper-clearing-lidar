library(dplyr)

wkdir <- "C:/Users/Cob/index/educational/usask/research/masters/data/hemispheres/"
csvout <- "hemi_lookup.csv"

folders <- list.files(path = wkdir, pattern = NULL, all.files = FALSE,
           full.names = FALSE, recursive = FALSE,
           ignore.case = FALSE, include.dirs = FALSE, no.. = FALSE)

lookup <- data.frame(folder = character(), filename = character())

for (ii in folders){
  path <- paste0(wkdir,ii,"/")
  filename <- list.files(path = path, pattern = NULL, all.files = FALSE,
                          full.names = FALSE, recursive = FALSE,
                          ignore.case = FALSE, include.dirs = FALSE, no.. = FALSE)
  tempdf <- data.frame(filename)
  tempdf$folder <- ii
  lookup <- rbind(lookup, tempdf)
}

lookup <- lookup %>%
  filter(filename != "Thumbs.db")

write.csv(lookup,paste0(wkdir,csvout), row.names = FALSE)
