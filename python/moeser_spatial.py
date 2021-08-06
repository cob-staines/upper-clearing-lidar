import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


# config


# canopy structure
df_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\merged_data_products\\merged_uf_r.25m_canopy_19_149_median-snow.csv'
df = pd.read_csv(df_in)

cc = df.loc[:, "lrs_cc"]
mdc = df.loc[:, "mean_dist_to_canopy"]
tga = df.loc[:, "total_gap_area"]

# constants (if testing for monotonicity, do these values matter?)
k = 0.3
p0 = 13.3
# precip = 3
precip = 8.6


###
def spr_from_mo(k=k, p0=p0, precip=precip):
    x1 = np.log10(mdc)
    x2 = np.log(cc)
    x3 = np.log(tga)

    imax = 2.167 * x1 - 3.410 * (x1 ** 2) + 55.761 * x2 + 181.858 * (x2 ** 2) - 2.493 * x3 + 0.499 * (x3 ** 2) + 20.819

    interception = imax / (1 + np.exp(-k * (precip - p0)))

    # correlations
    valid = ~np.isnan(interception) & ~np.isnan(df.loc[:, "dswe_fnsd_19_045-19_050"])
    spearman_storm_1 = spearmanr(df.loc[valid, "dswe_fnsd_19_045-19_050"], interception[valid])

    valid = ~np.isnan(interception) & ~np.isnan(df.loc[:, "dswe_fnsd_19_050-19_052"])
    spearman_storm_2 = spearmanr(df.loc[valid, "dswe_fnsd_19_050-19_052"], interception[valid])

    return (spearman_storm_1[0], spearman_storm_2[0])


def sensitivity_test(p_range):
    output = np.zeros((len(p_range), 2))
    for ii in range(0, len(p_range)):
        output[ii, :] = spr_from_mo(precip=p_range[ii])  # low sensitivity -- greater correlations for greater precip values (up to ~50) -- use observed values though
    return output

# p_range = np.linspace(0, 40, 50)
# out = sensitivity_test(p_range)
#
# plt.plot(p_range, out[:, 0])
# plt.plot(p_range, out[:, 1])

spr_from_mo()