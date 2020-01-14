library(dplyr)
library(ggplot2)

infile = 'C:/Users/Cob/index/educational/usask/research/masters/data/LiDAR/analysis/snow_depth/grid_histogram.csv'

data <- read.csv(infile)

try <- select(data, list(names(data)[5,7:297]))

foo <- reshape(data, 
               direction = "long",
               varying = list(names(data)[8:313]),
               v.names = "Value",
               idvar = "id",
               times = as.numeric(gsub("X", "", gsub("X.1", "-1", gsub("X.0", "-", names(data)[8:313])))))

foo <- foo[foo$id == 20 | foo$id == 200,]

ggplot(foo, aes(time, Value)) + 
  geom_path(aes(color = id))
