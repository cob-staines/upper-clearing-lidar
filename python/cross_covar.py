import rastools
import numpy as np
import pandas as pd
import tifffile as tif
from scipy.stats import spearmanr

batch_dir = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\ray_sampling\\batches\\lrs_hemi_uf_.25m_180px\\outputs\\'
# batch_dir = 'C:\\Users\\jas600\\workzone\\data\\hemigen\\mb_15_1m_pr.15_os10\\outputs\\'

batch_type = 'lin'  # 'exp', 'log', 'lin'

if batch_type == 'log':
    covar_out = batch_dir + "phi_theta_lookup_log_covar_training.csv"
    weighted_cv_out = batch_dir + "rshmetalog_log_weighted_cv.csv"
elif batch_type == 'exp':
    covar_out = batch_dir + "phi_theta_lookup_exp_covar_training.csv"
    weighted_cv_out = batch_dir + "rshmetalog_exp_weighted_cv.csv"
elif batch_type == 'lin':
    covar_out = batch_dir + "phi_theta_lookup_lin_covar_training.csv"
    weighted_cv_out = batch_dir + "rshmetalog_lin_weighted_cv.csv"

# scaling_coef = 0.166104  # all rings
scaling_coef = 0.194475  # dropping 5th ring

# load img meta
hemimeta = pd.read_csv(batch_dir + 'rshmetalog.csv')

imsize = hemimeta.img_size_px[0]

# merge with covariant
# var_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\mb_65\\dSWE\\19_045-19_050\\dswe_19_045-19_050_r.25m.tif'
# var_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\mb_65\\dSWE\\19_050-19_052\\dswe_19_050-19_052_r.25m.tif'
var_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\mb_65\\dSWE\\19_045-19_052\\dswe_19_045-19_052_r.25m.tif'
# var_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\mb_65\\dHS\\19_045-19_050\\dhs_19_045-19_050_r.25m.tif'
var = rastools.raster_to_pd(var_in, 'covariant')
hemi_var = pd.merge(hemimeta, var, left_on=('x_utm11n', 'y_utm11n'), right_on=('x_coord', 'y_coord'), how='inner')

# load angle template
angle_lookup = pd.read_csv(batch_dir + "phi_theta_lookup.csv")
phi = np.full((imsize, imsize), np.nan)
phi[(np.array(angle_lookup.x_index), np.array(angle_lookup.y_index))] = angle_lookup.phi * 180 / np.pi
max_phi = 90
# calculate radius to avoid pixels outside of circle
imrange = np.full((imsize, imsize), False)
imrange[phi <= max_phi] = True

# filter to desired images
#hemiList = hemi_swe.loc[(hemi_swe.swe.values >= 0) & (hemi_swe.swe.values <= 150), :]
# delineate training set and test set
set_param = np.random.random(len(hemi_var))  # ommit this line if only reloading the covariant
hemi_var.loc[:, 'training_set'] = set_param < .25
hemiList = hemi_var.loc[hemi_var.training_set, :].reset_index()


imstack = np.full([imsize, imsize, len(hemiList)], np.nan)
for ii in range(0, len(hemiList)):
    imstack[:, :, ii] = tif.imread(batch_dir + hemiList.file_name[ii])[:, :, 1] * scaling_coef
    print(str(ii + 1) + ' of ' + str(len(hemiList)))

if batch_type == 'log':
    imstack = np.log(imstack)
elif batch_type == 'exp':
    imstack = np.exp(-imstack)


# covar = np.full((imsize, imsize), np.nan)
corcoef = np.full((imsize, imsize), np.nan)
corcoef_e = np.full((imsize, imsize), np.nan)
corcoef_l = np.full((imsize, imsize), np.nan)
sprank = np.full((imsize, imsize), np.nan)

for ii in range(0, imsize):
    for jj in range(0, imsize):
        if imrange[jj, ii]:
            # covar[jj, ii] = np.cov(hemiList.covariant, imstack[jj, ii, :])[0, 1]
            corcoef[jj, ii] = np.corrcoef(hemiList.covariant, imstack[jj, ii, :])[0, 1]
            corcoef_e[jj, ii] = np.corrcoef(hemiList.covariant, np.exp(-21.613288 * imstack[jj, ii, :]))[0, 1]
            # corcoef_l[jj, ii] = np.corrcoef(hemiList.covariant, np.log(imstack[jj, ii, :]))[0, 1]
            # sprank[jj, ii] = spearmanr(hemiList.covariant, imstack[jj, ii, :])[0]

    print(ii)

########
# np.nanmax(corcoef_e)
# np.nanmin(corcoef_l)
# np.nanmin(corcoef)
# np.nanmin(sprank)
#
# np.where(corcoef_e == np.nanmax(corcoef_e))
# np.where(corcoef_l == np.nanmin(corcoef_l))
# np.where(corcoef == np.nanmin(corcoef))
# np.where(sprank == np.nanmin(sprank))
########



# methods for weighting
imstack_long = imstack
imstack_long[np.isnan(imstack_long)] = 0  # careful here..
imstack_long = np.swapaxes(np.swapaxes(imstack_long, 1, 2), 0, 1).reshape(imstack_long.shape[2], -1)


# # phi threshhold
# t_list = np.arange(0, 90, 5)
# w_data = pd.DataFrame(columns={"threshold", "corcoef", "sprank"})
#
# for tt in t_list:
#     weights = np.full((imsize, imsize), 0)
#     weights[phi <= tt] = 1
#     weights = weights / np.sum(weights)
#
#     w_stack = np.average(imstack, weights=weights.ravel(), axis=1)
#
#     w_corcoef = np.corrcoef(hemiList.covariant, w_stack)[0, 1]
#     w_sprank = spearmanr(hemiList.covariant, w_stack)[0]
#
#     new_row = {"threshold": tt,
#                "corcoef": w_corcoef,
#                "sprank": w_sprank}
#     w_data = w_data.append(new_row, ignore_index=True, verify_integrity=True)
#
# # phi dist from max
# max_coord = np.where(corcoef == np.nanmin(corcoef))
#
# yid, xid = np.indices((imsize, imsize))
# radist = np.sqrt((yid - max_coord[0][0]) ** 2 + (xid - max_coord[1][0]) ** 2) * np.pi / 180
#
# p_list = np.linspace(0, 20, 101)
# p_list = np.concatenate((p_list, np.linspace(25, 100, 16)))
# w_data = pd.DataFrame(columns={"power", "corcoef", "corcoef_e", "corcoef_l", "sprank"})
#
# for pp in p_list:
#     # normal distribution
#     weights = np.exp(- 0.5 * (pp * radist) ** 2)
#     weights[np.isnan(phi)] = 0
#     weights = weights ** pp
#
#     weights = weights / np.sum(weights)
#
#     w_stack = np.average(imstack_long, weights=weights.ravel(), axis=1)
#
#     w_corcoef = np.corrcoef(hemiList.covariant, w_stack)[0, 1]
#     w_corcoef_e = np.corrcoef(hemiList.covariant, np.exp(-20 * w_stack))[0, 1]
#     w_corcoef_l = np.corrcoef(hemiList.covariant, np.log(w_stack))[0, 1]
#     w_sprank = spearmanr(hemiList.covariant, w_stack)[0]
#
#     new_row = {"power": pp,
#                "corcoef": w_corcoef,
#                "corcoef_e": w_corcoef_e,
#                "corcoef_l": w_corcoef_l,
#                "sprank": w_sprank}
#     w_data = w_data.append(new_row, ignore_index=True, verify_integrity=True)
#
#     print(pp)
#
#
# # coef threshhold
# t_list = np.linspace(np.nanmin(corcoef), 0, 20)
# w_data = pd.DataFrame(columns={"threshold", "corcoef", "sprank", "count"})
#
# for tt in t_list:
#     weights = np.full((imsize, imsize), 0)
#     weights[corcoef <= tt] = 1
#     count = np.sum(weights)
#     weights = weights / np.sum(weights)
#
#     w_stack = np.average(imstack_long, weights=weights.ravel(), axis=1)
#
#     w_corcoef = np.corrcoef(hemiList.covariant, w_stack)[0, 1]
#     w_sprank = spearmanr(hemiList.covariant, w_stack)[0]
#
#     new_row = {"threshold": tt,
#                "corcoef": w_corcoef,
#                "sprank": w_sprank,
#                "count": count}
#     w_data = w_data.append(new_row, ignore_index=True, verify_integrity=True)
#
# # weighted coef power
# w_data = pd.DataFrame(columns={"power", "corcoef", "sprank"})
# p_list = np.linspace(3, 10, 20)
# ii = 0
# for pp in p_list:
#     weights = np.full((imsize, imsize), 0.0)
#     weights[corcoef_e >= 0] = corcoef_e[corcoef_e >= 0] ** pp
#     #weights[~np.isnan(corcoef_e)] = corcoef_e[~np.isnan(corcoef_e)] ** pp
#     # weights[corcoef <= tt] = corcoef[corcoef <= tt] ** 2
#     # count = np.sum(corcoef <= tt)
#     weights = weights / np.nansum(np.abs(weights))
#
#     w_stack = np.average(imstack_long, weights=weights.ravel(), axis=1)
#
#     w_corcoef = np.corrcoef(hemiList.covariant, w_stack)[0, 1]
#     w_sprank = spearmanr(hemiList.covariant, w_stack)[0]
#
#     new_row = {"power": pp,
#                "corcoef": w_corcoef,
#                "sprank": w_sprank}
#     w_data = w_data.append(new_row, ignore_index=True, verify_integrity=True)
#
#     print(ii)
#     ii += 1
#
#
# # weighted sprank threshhold
# t_list = np.linspace(np.nanmin(sprank), 0, 20)
# w_data = pd.DataFrame(columns={"threshold", "corcoef", "sprank", "count"})
#
# for tt in t_list:
#     weights = np.full((imsize, imsize), 0.0)
#     weights[sprank <= tt] = -sprank[sprank <= tt]
#     count = np.sum(sprank <= tt)
#     weights = weights / np.sum(weights)
#
#     w_stack = np.average(imstack_long, weights=weights.ravel(), axis=1)
#
#     w_corcoef = np.corrcoef(hemiList.covariant, w_stack)[0, 1]
#     w_sprank = spearmanr(hemiList.covariant, w_stack)[0]
#
#     new_row = {"threshold": tt,
#                "corcoef": w_corcoef,
#                "sprank": w_sprank,
#                "count": count}
#     w_data = w_data.append(new_row, ignore_index=True, verify_integrity=True)
#
#
# # optimize exp scalar
# c_list = np.linspace(1, 50, 300)
# c_data = pd.DataFrame(columns={"scalar", "corcoef"})
#
# for cc in c_list:
#     corcoef = np.corrcoef(hemiList.covariant, np.exp(-cc * w_stack))[0, 1]
#
#     new_row = {"scalar": cc,
#                "corcoef": corcoef}
#     c_data = c_data.append(new_row, ignore_index=True, verify_integrity=True)
#
# plt.plot(c_data.scalar, c_data.corcoef)


# optimization of field of view and interaction number
from scipy.optimize import fmin_bfgs


def gbgf(x0):
    sig = x0[0]
    intnum = x0[1]
    cpy = x0[2]
    cpx = x0[3]

    yid, xid = np.indices((imsize, imsize))
    radist = np.sqrt((yid - cpy) ** 2 + (xid - cpx) ** 2) * np.pi / 180

    weights = np.exp(- 0.5 * (radist / sig) ** 2)  # gaussian
    weights[np.isnan(phi)] = 0
    weights = weights / np.sum(weights)

    w_stack = np.average(imstack_long, weights=weights.ravel(), axis=1)

    valid = (hemiList.covariant > 0) & (hemiList.covariant <=50)

    # w_corcoef_e = np.corrcoef(hemiList.covariant, np.exp(-intnum * w_stack))[0, 1]
    w_corcoef_e = np.corrcoef(hemiList.covariant[valid], np.exp(-intnum * w_stack[valid]))[0, 1]

    return -w_corcoef_e

Nfeval = 1
def callbackF(Xi):
    global Nfeval
    print('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}   {5: 3.6f}'.format(Nfeval, Xi[0], Xi[1], Xi[2], Xi[3], gbgf(Xi)))
    Nfeval += 1

print('{0:4s}   {1:9s}   {2:9s}   {3:9s}   {4:9s}   {5:9s}'.format('Iter', ' X1', ' X2', 'X3', 'X4', 'f(X)'))
x0 = np.array([0.11, 21.6, 96, 85], dtype=np.double)

[xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflg] = \
    fmin_bfgs(gbgf, x0, callback=callbackF, maxiter=25, full_output=True, retall=False)
# xopt = np.array([0.1296497, 21.57953188, 96.95887751, 86.24391083])  # gaussian optimization,
# fopt = -0.4744932

sig_out = xopt[0] * 180 / np.pi
intnum_out = 1/xopt[1]
phi_out = np.sqrt((xopt[2] - (imsize - 1)/2) ** 2 + (xopt[3] - (imsize - 1)/2) ** 2)
theta_out = np.arctan2(-(xopt[3] - (imsize - 1)/2), -(xopt[2] - (imsize - 1)/2)) * 180 / np.pi




####### Visualization
import matplotlib
matplotlib.use('Qt5Agg')
# matplotlib.use('Agg')
import matplotlib.pyplot as plt


## scatter plot
x0 = xopt

sig = x0[0]
intnum = x0[1]
cpy = x0[2]
cpx = x0[3]

yid, xid = np.indices((imsize, imsize))
radist = np.sqrt((yid - cpy) ** 2 + (xid - cpx) ** 2) * np.pi / 180

weights = np.exp(- 0.5 * (radist / sig) ** 2)
weights[np.isnan(phi)] = 0
weights = weights / np.sum(weights)

w_stack = np.average(imstack_long, weights=weights.ravel(), axis=1)

fig = plt.figure()
fig.subplots_adjust(top=0.90, bottom=0.12, left=0.12)
ax1 = fig.add_subplot(111)
ax1.set_title('dSWE vs. directionally weighted snowfall transmission <$T_s$>\n Upper Forest, 14-21 Feb. 2019, 25cm resolution')
ax1.set_xlabel("dSWE (mm)")
ax1.set_ylabel("<$T_s$> [-]")
plt.scatter(hemi_var.covariant[hemi_var.training_set.values], np.exp(-intnum * w_stack), s=2, alpha=.25)
plt.xlim(-50, 100)

# ## calculate interaction scalar across hemisphere
# plt.scatter(np.log(hemi_var.covariant[hemi_var.training_set.values]), -w_stack, s=2, alpha=.25)
# np.nanmean(-np.log(hemi_var.covariant[hemi_var.training_set.values]) / w_stack)
#
# from scipy.optimize import curve_fit
#
# corcoef_each = np.full((imsize, imsize), np.nan)
# is_each = np.full((imsize, imsize), np.nan)
# gg_each = np.full((imsize, imsize), np.nan)
#
# for ii in range(0, imsize):
#     for jj in range(0, imsize):
#         if imrange[jj, ii]:
#             def expfunc(p1):
#                 return -np.corrcoef(hemi_var.covariant[hemi_var.training_set.values], np.exp(-p1 * imstack[jj, ii, :]))[0, 1]
#
#
#             [popt, ffopt, ggopt, BBopt, func_calls, grad_calls, warnflg] = \
#                 fmin_bfgs(expfunc,
#                           1,
#                           maxiter=100,
#                           full_output=True,
#                           retall=False)
#
#             is_each[jj, ii] = popt[0]
#             corcoef_each[jj, ii] = -ffopt
#             gg_each[jj, ii] = ggopt
#
#     print(ii)
#
# is_each_qc = is_each.copy()
# is_each_qc[gg_each > .00001] = np.nan
# is_each_qc[is_each_qc < 0] = np.nan
#
# plt.imshow(is_each_qc)
# plt.imshow(gg_each)
# corcoef_each_qc = corcoef_each.copy()
# corcoef_each_qc[corcoef_each_qc == 1] = np.nan
# plt.imshow(corcoef_each_qc)
# plt.colorbar()


# Here you give the initial parameters for p0 which Python then iterates over
# to find the best fit
jj = 90
ii = 90

popt, pcov = curve_fit(expfunc, w_stack, hemi_var.covariant[hemi_var.training_set.values], p0=(20.0), bounds=(0, np.inf))

plt.scatter(hemi_var.covariant[hemi_var.training_set.values], np.exp(-popt[0] * w_stack), s=2, alpha=.25)

## plot optimization topography

# sigma

# sample optimization topography for sigma

v_list = np.linspace(0.001, np.pi/2, 200)  # sig
w_data = pd.DataFrame(columns={"sig", "intnum", "cpy", "cpx", "corcoef_e"})
ii = 0
for vv in v_list:
    # x = np.array([0.1296497, 21.57953188, 96.95887751, 86.24391083])
    xx = np.array([vv, 21.57953188, 96.95887751, 86.24391083])

    w_corcoef_e = -gbgf(xx)

    new_row = {"sig": xx[0],
               "intnum": xx[1],
               "cpy": xx[2],
               "cpx": xx[3],
               "corcoef_e": w_corcoef_e}
    w_data = w_data.append(new_row, ignore_index=True, verify_integrity=True)

    ii += 1
    print(ii)

fig = plt.figure()
fig.subplots_adjust(top=0.90, bottom=0.15, left=0.15)
ax1 = fig.add_subplot(111)
ax1.set_title('Optimization of Gaussian weight function width $\sigma$\nfor modeling snow accumulation')
ax1.set_ylabel("Pearson's Correlation Coeficient\ndSWE vs. Directionally Weighted Snowfall Transmission")
ax1.set_xlabel("Standard deviation of Gausian weight function $\sigma$ [$^{\circ}$]")
plt.plot(w_data.sig * 180 / np.pi, w_data.corcoef_e)

# interaction scalar

# sample optimization topography for gamma

v_list = np.linspace(1, 100, 100)  # intnum
w_data = pd.DataFrame(columns={"sig", "intnum", "cpy", "cpx", "corcoef_e"})
ii = 0
for vv in v_list:
    # x = np.array([0.1296497, 21.57953188, 96.95887751, 86.24391083])
    xx = np.array([0.1296497, vv, 96.95887751, 86.24391083])

    w_corcoef_e = -gbgf(xx)

    new_row = {"sig": xx[0],
               "intnum": xx[1],
               "cpy": xx[2],
               "cpx": xx[3],
               "corcoef_e": w_corcoef_e}
    w_data = w_data.append(new_row, ignore_index=True, verify_integrity=True)

    ii += 1
    print(ii)

fig = plt.figure()
fig.subplots_adjust(top=0.90, bottom=0.15, left=0.15)
ax1 = fig.add_subplot(111)
ax1.set_title('Optimization of snowfall transmission interaction factor <$\gamma$>\nfor modeling snow accumulation')
ax1.set_ylabel("Pearson's Correlation Coeficient\ndSWE vs. Directionally Weighted Snowfall Transmission")
ax1.set_xlabel("Interaction factor <$\gamma$> [-]")
plt.plot(w_data.intnum, w_data.corcoef_e)


## plot hemispherical footprint

# rerun corcoef_e with optimized transmission scalar
corcoef_e = np.full((imsize, imsize), np.nan)
for ii in range(0, imsize):
    for jj in range(0, imsize):
        if imrange[jj, ii]:
            corcoef_e[jj, ii] = np.corrcoef(hemiList.covariant, np.exp(-xopt[1] * imstack[jj, ii, :]))[0, 1]

    print(ii)

# prep divergent colormap
import matplotlib.colors as colors
# set the colormap and centre the colorbar

class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...

        # maintain equal intensity scaling on either side of midpoint
        dmid = (self.midpoint - self.vmin, self.vmax - self.midpoint)
        maxmid = np.max(dmid)
        r_min = self.midpoint - maxmid
        r_max = self.midpoint + maxmid
        c_low = (self.vmin - r_min) / (2 * maxmid)
        c_high = 1 + (r_max - self.vmax) / (2 * maxmid)
        x, y = [self.vmin, self.midpoint, self.vmax], [c_low, 0.5, c_high]

        # use vmin and vmax as extreme ends of colormap
        # x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]

        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))



# colormap parameters
set = corcoef_e  # this is what we are plotting
val_min = np.nanmin(set)
val_mid = 0
val_max = np.nanmax(set)
cmap = matplotlib.cm.RdBu

# main plot
fig = plt.figure(figsize=(7, 7))
fig.suptitle("Correlation of dSWE with snowfall transmission over hemisphere\nUpward-looking, Upper Forest, 14-21 Feb 2019, 25cm resolution")
#create axes in the background to show cartesian image
ax0 = fig.add_subplot(111)
im = ax0.imshow(set, cmap=cmap, clim=(val_min, val_max), norm=MidpointNormalize(vmin=-abs_max, midpoint=val_mid, vmax=abs_max))
ax0.axis("off")

# create polar axes and labels
ax = fig.add_subplot(111, polar=True, label="polar")
ax.set_facecolor("None")
ax.set_rmax(90)
ax.set_rgrids(np.linspace(0, 90, 7), labels=['', '15$^\circ$', '30$^\circ$', '45$^\circ$', '60$^\circ$', '75$^\circ$', '90$^\circ$'], angle=315)
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
# # ax.set_thetagrids(np.linspace(0, 360, 8, endpoint=False), labels=['N', '', 'W', '', 'S', '', 'E', ''])
ax.set_thetagrids(np.linspace(0, 360, 4, endpoint=False), labels=['N\n  0$^\circ$', 'W\n  270$^\circ$', 'S\n  180$^\circ$', 'E\n  90$^\circ$'])

# add colorbar
fig.subplots_adjust(top=0.95, left=0.1, right=0.75, bottom=0.05)
cbar_ax = fig.add_axes([0.85, 0.20, 0.03, 0.6])
fig.colorbar(im, cax=cbar_ax)
cbar_ax.set_ylabel("Pearson's correlation coefficient")
# cbar_ax.set_label("Pearson's Correlation Coefficient", rotation=270)

# contours
ax.set_rgrids([])  # no rgrids
ax.grid(False)  # no grid
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
matplotlib.rcParams["lines.linewidth"] = 1
CS = ax0.contour(set, np.linspace(-.4, .4, 9), colors="k")
plt.clabel(CS, inline=1, fontsize=8)

###
# plot radial distribution of corcoef_e
sig = xopt[0]
cpy = xopt[2]
cpx = xopt[3]
yid, xid = np.indices((imsize, imsize))
radist = np.sqrt((yid - cpy) ** 2 + (xid - cpx) ** 2) * np.pi / 180

radcor = pd.DataFrame({"phi": np.rint(np.ravel(radist) * 180 / np.pi).astype(int),
                       "corcoef_e": np.ravel(corcoef_e)})

radmean = radcor.groupby('phi').mean().reset_index(drop=False)
radmean = radmean.assign(gaus=np.exp(- 0.5 * (radmean.phi * np.pi / (sig * 180)) ** 2))

plt.plot(radmean.phi, radmean.corcoef_e)
plt.plot(radmean.phi, radmean.gaus * np.nanmax(radmean.corcoef_e))
plt.plot(radmean.phi, radmean.corcoef_e - radmean.gaus * np.nanmax(radmean.corcoef_e))

###

maxcoords = np.where(corcoef_e == np.nanmax(corcoef_e))
jj = maxcoords[0][0]
ii = maxcoords[1][0]

plt.scatter(hemiList.covariant, np.exp(-xopt[1] * imstack[jj, ii, :]), alpha=0.1, s=5)

immid = ((imsize - 1)/2).astype(int)
jj = immid
ii = immid
plt.scatter(hemiList.covariant, np.exp(-xopt[1] * imstack[jj, ii, :]), alpha=0.1, s=5)
# cumulative weights... this is all messed

plt.imshow(sprank, cmap=plt.get_cmap('Purples_r'))
plt.colorbar()


# angle_lookup_covar.sqr_covar_weighted
# angle_lookup_covar.sort_values(phi)
# peace = angle_lookup_covar.loc[~np.isnan(angle_lookup_covar.covar), :]
# peace.loc[:, 'sqr_covar_weight_cumsum'] = np.cumsum(peace.sort_values('phi').sqr_covar_weight.values).copy()
#
# plt.scatter(peace.phi, peace.sqr_covar_weight_cumsum)