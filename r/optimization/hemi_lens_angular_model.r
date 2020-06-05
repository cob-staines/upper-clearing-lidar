library('ggplot2')

infile = 'C:/Users/Cob/index/educational/usask/research/masters/data/hemispheres/calibration/angular_measurements.csv'
data = read.csv(infile, header=TRUE, na.strings = c("NA",""), sep=",")

rad = data$dist[7]
data$R = data$dist/rad
data$theta_rad = data$theta*pi/180

ggplot(data, aes(x=theta_rad, y=R)) + 
  geom_point() +
  geom_abline(intercept = 0, slope = 2/pi, color="red", size=1.5)

models.lm = lm(R ~ theta_rad, data=data)

summary(models.lm)$r.squared
