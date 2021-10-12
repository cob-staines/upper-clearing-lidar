import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


# config
plot_out_dir = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\graphics\\thesis_graphics\\modeling snow accumulation\\moeser\\"

# canopy structure
df_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\merged_data_products\\merged_uf_r.25m_canopy_19_149_median-snow.csv'
df = pd.read_csv(df_in)

cc = df.loc[:, "lrs_cc"]
mdc = df.loc[:, "mean_dist_to_canopy_1.25"]
tga = df.loc[:, "total_gap_area_1.25"]

# constants (if testing for monotonicity, do these values matter?)
k = 0.3
p0 = 13.3



###
# def spr_from_mo(k=k, p0=p0, precip=precip):
x1 = np.log(mdc)
x2 = np.log(cc)
x3 = np.log(tga)

imax = 2.167 * x1 - 3.410 * (x1 ** 2) + 55.761 * x2 + 181.858 * (x2 ** 2) - 2.493 * x3 + 0.499 * (x3 ** 2) + 20.819

# correlations
precip = 105.06450584
swevar = df.loc[:, "swe_fcon_19_052"]
interception = imax / (1 + np.exp(-k * (precip - p0)))
throughfall = precip - interception
valid = ~np.isnan(interception) & ~np.isnan(swevar)
spearman_swe = spearmanr(swevar[valid], throughfall[valid])

fig = plt.figure()
fig.subplots_adjust(top=0.90, bottom=0.12, left=0.12)
ax1 = fig.add_subplot(111)
ax1.set_title('SWE vs. Moeser et al. throughfall $F_{M}$\n Forest, 21 Feb. 2019')
ax1.set_xlabel("SWE [mm]")
ax1.set_ylabel("$F_{M}$ [mm]")
plt.ylim(np.nanquantile(throughfall, .005) - 2, np.nanquantile(throughfall, .995) + 2)
plt.xlim(np.nanquantile(swevar, .005) - 2, np.nanquantile(swevar, .995) + 2)
plt.scatter(swevar[valid], throughfall[valid], alpha=0.10, s=2)
mm_mod = np.array([np.nanmin(swevar), np.nanmax(swevar)])
# plt.plot(mm_mod, mm_mod, c='Black', linewidth=1)
fig.savefig(plot_out_dir + "SWE vs moeser 19_052.png")


# precip = 3  # old
precip = 9.03
swevar = df.loc[:, "dswe_fnsd_19_045-19_050"]
interception = imax / (1 + np.exp(-k * (precip - p0)))
throughfall = precip - interception
valid = ~np.isnan(interception) & ~np.isnan(swevar)
spearman_storm_1 = spearmanr(swevar[valid], throughfall[valid])

fig = plt.figure()
fig.subplots_adjust(top=0.90, bottom=0.12, left=0.12)
ax1 = fig.add_subplot(111)
ax1.set_title('$\Delta$SWE vs. Moeser et al. throughfall $F_{M}$\n Forest, 14-19 Feb. 2019')
ax1.set_xlabel("$\Delta$SWE [mm]")
ax1.set_ylabel("$F_{M}$ [mm]")
plt.ylim(np.nanquantile(throughfall, .005) - 0.05, np.nanquantile(throughfall, .995) + 0.05)
plt.xlim(np.nanquantile(swevar, .005) - 0.05, np.nanquantile(swevar, .995) + 0.05)
plt.scatter(swevar[valid], throughfall[valid], alpha=0.10, s=2)
mm_mod = np.array([np.nanmin(swevar), np.nanmax(swevar)])
# plt.plot(mm_mod, mm_mod, c='Black', linewidth=1)
fig.savefig(plot_out_dir + "SWE vs moeser 045-050.png")

# precip = 8.6  # old
precip = 6.85
swevar = df.loc[:, "dswe_fnsd_19_050-19_052"]
interception = imax / (1 + np.exp(-k * (precip - p0)))
throughfall = precip - interception
valid = ~np.isnan(interception) & ~np.isnan(swevar)
spearman_storm_2 = spearmanr(swevar[valid], throughfall[valid])

fig = plt.figure()
fig.subplots_adjust(top=0.90, bottom=0.12, left=0.12)
ax1 = fig.add_subplot(111)
ax1.set_title('$\Delta$SWE vs. Moeser et al. throughfall $F_{M}$\n Forest, 19-21 Feb. 2019')
ax1.set_xlabel("$\Delta$SWE [mm]")
ax1.set_ylabel("$F_{M}$ [mm]")
plt.ylim(np.nanquantile(throughfall, .005) - 0.25, np.nanquantile(throughfall, .995) + 0.25)
plt.xlim(np.nanquantile(swevar, .005) - 0.25, np.nanquantile(swevar, .995) + 0.25)
plt.scatter(swevar[valid], throughfall[valid], alpha=0.10, s=2)
mm_mod = np.array([np.nanmin(swevar), np.nanmax(swevar)])
# plt.plot(mm_mod, mm_mod, c='Black', linewidth=1)
fig.savefig(plot_out_dir + "SWE vs moeser 050-052.png")

    # return (spearman_swe, spearman_storm_1[0], spearman_storm_2[0])


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

# optimization

from scipy.optimize import fmin_bfgs, minimize

# valid = ~np.isnan(interception) & ~np.isnan(df.loc[:, "dswe_fnsd_19_050-19_052"])
swevar = df.loc[:, "swe_fcon_19_052"]
precip = 105.06450584

# swevar = df.loc[:, "dswe_fnsd_19_045-19_050"]
# precip = 9.03

# swevar = df.loc[:, "dswe_fnsd_19_050-19_052"]
# precip = 6.85
valid = ~np.isnan(swevar)

k = 0.3
p0 = 13.3
def mo_accum(x0):

    v0, v1, v2, v3, v4, v5, v6 = x0

    x1 = np.log10(mdc)
    x2 = np.log(cc)
    x3 = np.log(tga)

    imax = v0 * x1 - v1 * (x1 ** 2) + v2 * x2 + v3 * (x2 ** 2) - v4 * x3 + v5 * (x3 ** 2) + v6

    interception = imax / (1 + np.exp(-k * (precip - p0)))
    accumulation = precip - interception

    ssres = np.sum((swevar - accumulation) ** 2)

    return ssres


def rsq(x0):
    ssres = mo_accum(x0)

    # dswe = covariant
    dswe = swevar[valid]

    sstot = np.sum((dswe - np.mean(dswe)) ** 2)
    return 1 - ssres / sstot

def mo_accum_data(x0):
    # s_bar, h_bar, l_0, snow_dens, precip = x0

    v0, v1, v2, v3, v4, v5, v6 = x0

    x1 = np.log10(mdc)
    x2 = np.log(cc)
    x3 = np.log(tga)

    imax = v0 * x1 - v1 * (x1 ** 2) + v2 * x2 + v3 * (x2 ** 2) - v4 * x3 + v5 * (x3 ** 2) + v6

    interception = imax / (1 + np.exp(-k * (precip - p0)))
    accumulation = precip - interception

    # ssres = np.sum((swevar - accumulation) ** 2)

    return accumulation

x0 = np.array([ -30.63362723, 22.79966982, 9.71420853, -197.25336491, 4.02806702, 0.85775479, 126.01717906])  # 19_052 , p_s = 0.599390890905355
x0 = np.array([-74.01782945, -19.14164369, 9.33915791, 3.21992909, -7.88632607, -0.3427642, 59.16903417])  # storm 1 , p_s = 0.2590593393981892
x0 = np.array([102.09990454,   42.84472146, -21.7541577, -125.07498835, 14.17425099, 0.95957025, 8.94397249])  # storm 2 , p_s = 0.21996477019477148

res = minimize(mo_accum, x0, method='Nelder-Mead', options={'maxiter': 20000})


throughfall = mo_accum_data(res.x)

plt.scatter(swevar, throughfall, s=2, alpha=0.05)
spearmanr(swevar[valid], throughfall[valid])
