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

lai = df.loc[:, "lrs_lai_2000"]
# lai = df.loc[:, "lrs_lai_60_deg"]
# lai = df.loc[:, "lrs_lai_15_deg"]
# lai = df.loc[:, "lrs_lai_1_deg"]
cc = df.loc[:, "lrs_cc"]
chm = df.loc[:, "chm_25"]

# constants (if testing for monotonicity, do these values matter?)
# precip = 3  # mm swe
precip = 8.6  # mm swe
snow_dens = 68  # kg/m^3
s_bar = 5.9  # kg/m^2
h_bar = np.nanmax(chm)
l_0 = 0

###
def spr_from_hp(precip=precip, snow_dens=snow_dens, s_bar=s_bar, h_bar=h_bar, l_0 = l_0):
    ss = s_bar * (0.27 + 46 / snow_dens)
    l_star = ss * lai
    cp = cc / (1 - (cc * chm / h_bar))
    kk = cp / l_star
    ##

    interception = (l_star - l_0) * (1 - np.exp(-kk * precip))

    # correlations
    valid = ~np.isnan(interception) & ~np.isnan(df.loc[:, "dswe_fnsd_19_045-19_050"])
    spearman_storm_1 = spearmanr(df.loc[valid, "dswe_fnsd_19_045-19_050"], interception[valid])

    valid = ~np.isnan(interception) & ~np.isnan(df.loc[:, "dswe_fnsd_19_050-19_052"])
    spearman_storm_2 = spearmanr(df.loc[valid, "dswe_fnsd_19_050-19_052"], interception[valid])

    return (spearman_storm_1, spearman_storm_2)


def sensitivity_test(p_range):
    output = np.zeros((len(p_range), 2))
    for ii in range(0, len(p_range)):
        output[ii, :] = spr_from_hp(precip=p_range[ii])  # low sensitivity -- greater correlations for greater precip values (up to ~50) -- use observed values though
        # output[ii, :] = spr_from_hp(snow_dens=p_range[ii])  # no sensitivity -- use observed value
        # output[ii, :] = spr_from_hp(s_bar=p_range[ii])  # low sensitivity -- use textbook value
        # output[ii, :] = spr_from_hp(h_bar=p_range[ii])  # high sensitivity -- less sensitive for values around max canopy height
        # output[ii, :] = spr_from_hp(l_0=p_range[ii])  # moderate sensitivity -- greatest correlation around 0 -- use 0

    return output

# p_range = np.linspace(0, 40, 50)
# out = sensitivity_test(p_range)
#
# plt.plot(p_range, out[:, 0])
# plt.plot(p_range, out[:, 1])

spr_from_hp()