library('ggplot2')
library('BAMMtools')

trees_in = "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/19_149/19_149_snow_off/OUTPUT_FILES/DFT/19_149_all_200311_628000_564652_chm_.25m_trickle_treetops.csv"
trees = read.csv(trees_in, header=TRUE, na.strings = c("NA",""), sep=",")

ggplot(trees, aes(x=height_m, y=area_m2)) +
  geom_point()

pw_fun <- function(x) log10(3.75105 - 0.17919*x + 0.01241*x^2)
cob_fun <- function(x) log10(1 - 0.17919*x + 0.01241*x^2)
cob_2fun <- function(x) log10(0.009134*x^2 - 0.1554*x + 1.275)
cob_3fun <- function(x) log10(0.009740*x^2 - 0.1396*x + 1.403)

pw_fun <- function(x) 3.75105 - 0.17919*x + 0.01241*x^2
cob_fun <- function(x) 1 - 0.17919*x + 0.01241*x^2
cob_2fun <- function(x) 0.009134*x^2 - 0.1554*x + 1.275
cob_3fun <- function(x) 0.009740*x^2 - 0.1396*x + 1.403

iso_in = "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/19_149/19_149_snow_off/OUTPUT_FILES/DFT/19_149_all_200311_628000_5646525_spike_free_chm_.10m_prom_treetops.csv"
iso = read.csv(iso_in, header=TRUE, na.strings = c("NA",""), sep=",")
# calculate prominence-height ratio
iso$prom_over_h = iso$prom_expected/iso$elev

# threshold by watershed area
iso_filtered = iso[iso$area_pixels > 1,]
iso_filtered = iso_filtered[iso_filtered$prom_over_h >= 0.5,]

# unfiltered plots
ggplot(iso, aes(x=elev, y=iso_expected, color = prom_over_h)) +
  geom_point() +
  stat_function(fun = pw_fun) +
  stat_function(fun = cob_fun) +
  ylim(0, 10) +
  scale_y_log10() +
  labs(title="CHM heigh vs. isolation for local maxima", x="Canopy height (m)", y="isolation (m)", color="prominence/height")

ggplot(iso, aes(x=prom_expected, y=iso_expected, color=elev)) +
  geom_point() +
  ylim(0, 10)

ggplot(iso, aes(x=elev, y=prom_expected, color=iso_expected)) +
  geom_point()

# filtered plots
ggplot(iso_filtered, aes(x=elev, y=iso_expected, color = prom_over_h)) +
  geom_point() +
  stat_function(fun = pw_fun) +
  ylim(0, 10) +
  labs(title="CHM heigh vs. isolation for local maxima", x="Canopy height (m)", y="isolation (m)", color="prominence/height")

# other plots
ggplot(iso, aes(x=elev, y=iso_expected, color = prom_over_h)) +
  geom_point() +
  xlim(4, 30) +
  ylim(0, 5) +
  stat_function(fun = pw_fun) 
  stat_function(fun = cob_2fun) +
  stat_function(fun = cob_3fun)

ggplot(iso, aes(x=elev, y=sqrt(area_m2/pi))) +
  stat_density_2d(aes(fill = ..level..), geom = "polygon")

ggplot(iso, aes(x=prom_expected, y=iso_expected, color=area_radius_over_h)) +
  geom_point() +
  scale_y_log10()

ggplot(iso, aes(x=prom_expected, y=iso_expected, color=elev)) +
  geom_point() +
  scale_y_log10()

ggplot(iso, aes(x=elev, y=prom_expected, color=iso_expected)) +
  geom_point() +
  scale_y_log10()

kho_in = "C:/Users/Cob/index/educational/usask/research/masters/data/lidar/19_149/19_149_snow_off/OUTPUT_FILES/DFT/19_149_all_200311_628000_5646525_spike_free_chm_.10m_kho_treetops.csv"
kho = read.csv(kho_in, header=TRUE, na.strings = c("NA",""), sep=",")
kho$true_peak = as.logical(kho$true_peak)
merged = merge(kho, iso, by=c("UTM11N_x", "UTM11N_y"), all = FALSE, suffixes=c("_kho", "_prom"))

# just to show heights are identical
ggplot(data=merged, aes(x=peak_z, y=elev)) +
  geom_point()

# looking for good predictors of true vs false peaks

# watershed area -> no
ggplot(data=merged, aes(x=area_m2_kho, y=area_m2_prom)) +
  geom_point()

# peak height -> standalone no
ggplot(data=merged, aes(x=area_m2_kho, y=peak_z)) +
  geom_point()

# peak prominence -> standalone no
ggplot(data=merged, aes(x=area_m2_kho, y=prom_expected)) +
  geom_point()

# prom_over_h -> standalone no
ggplot(data=merged, aes(x=area_m2_kho, y=prom_over_h)) +
  geom_point()

# isolation -> perhaps. not as clear as kho. Do we gain explanatory power?
ggplot(data=merged, aes(x=area_m2_kho, y=iso_expected)) +
  geom_point() +
  scale_y_log10()


# try kmeans
peace = kmeans(merged$area_m2_kho, 2)

df = data.frame(merged$area_m2_kho, merged$peak_z, log10(merged$iso_expected))
peace = kmeans(df, 2)

# try jenks natural breaks
peace = getJenksBreaks(merged$area_m2_kho, 1)

merged$cluster = peace$cluster

ggplot(data=merged, aes(x=area_m2_kho, y=iso_expected, color=cluster)) +
  geom_point() +
  scale_y_log10()


ggplot(data=merged, aes(x=elev, y=iso_expected)) +
  geom_point(aes(color = true_peak, alpha=I(0.3))) +
  stat_function(fun = pw_fun, aes(size="minmum threshold")) +
  scale_y_log10() +
  scale_color_manual(values=c("#777777", "red")) +
  labs(title="Height vs. isolation for CHM local maxima", x="height above ground (m)", y="isolation -- min distance to higher object (m)", color="True peak", size="Popescu & Wynne 2004")

ggplot(data=kho, aes(kho$area_m2)) +
  geom_histogram(binwidth = 0.1) +
  scale_y_log10() +
  labs(title="Histogram of Khosravipour 2015 peak neighborhood area", x="Peak neighborhood area (m^2)", y="count")

