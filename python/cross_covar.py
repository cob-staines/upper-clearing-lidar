import rastools
import numpy as np
import pandas as pd
import tifffile as tif

batch_dir = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\ray_sampling\\batches\\lrs_hemi_uf_.25m_180px\\outputs\\'
# batch_dir = 'C:\\Users\\jas600\\workzone\\data\\hemigen\\mb_15_1m_pr.15_os10\\outputs\\'

# # output files
# covar_out = batch_dir + "phi_theta_lookup_covar_training.csv"
# weighted_cv_out = batch_dir + "rshmetalog_weighted_cv.csv"

# scaling coefficient converts from expected returns to expected contact number
# scaling_coef = 0.166104  # all rings
scaling_coef = 0.194475  # dropping 5th ring

# load img meta
hemimeta = pd.read_csv(batch_dir + 'rshmetalog.csv')
imsize = hemimeta.img_size_px[0]

# load covariant
# var_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\mb_65\\dSWE\\19_045-19_050\\dswe_19_045-19_050_r.25m.tif'
var_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\mb_65\\dSWE\\19_050-19_052\\dswe_19_050-19_052_r.25m.tif'
var = rastools.raster_to_pd(var_in, 'covariant')
# merge with image meta
hemi_var = pd.merge(hemimeta, var, left_on=('x_utm11n', 'y_utm11n'), right_on=('x_coord', 'y_coord'), how='inner')

# load image pixel angle lookup
angle_lookup = pd.read_csv(batch_dir + "phi_theta_lookup.csv")
# build phi image (in radians)
phi = np.full((imsize, imsize), np.nan)
phi[(np.array(angle_lookup.x_index), np.array(angle_lookup.y_index))] = angle_lookup.phi
# build theta image (in radians)
theta = np.full((imsize, imsize), np.nan)
theta[(np.array(angle_lookup.x_index), np.array(angle_lookup.y_index))] = angle_lookup.theta

# limit analysis by phi (in radians)
max_phi = 90 * (np.pi / 180)
# calculate radius range to avoid pixels outside of circle
imrange = np.full((imsize, imsize), False)
imrange[phi <= max_phi] = True

# # filter hemimeta to desired images
# THIS WOULD BE WHERE WE BIAS CORRECT ACCORDING TO A CERTAIN DISTRIBUTION
# delineate training set (set_param < param_thresh) and test set (set_param >= param thresh)
param_thresh = 0.25
set_param = np.random.random(len(hemi_var))
hemi_var.loc[:, 'training_set'] = set_param < param_thresh
# build hemiList from training_set only
hemiList = hemi_var.loc[hemi_var.training_set, :].reset_index()

# load hemiList images to imstack
imstack = np.full([imsize, imsize, len(hemiList)], np.nan)
for ii in range(0, len(hemiList)):
    imstack[:, :, ii] = tif.imread(batch_dir + hemiList.file_name[ii])[:, :, 1] * scaling_coef
    print(str(ii + 1) + ' of ' + str(len(hemiList)))

# preview of correlation coefficient
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



# reshape imstack (as imstack_long) for optimization
imstack_long = imstack
imstack_long[np.isnan(imstack_long)] = 0  # careful here..
imstack_long = np.swapaxes(np.swapaxes(imstack_long, 1, 2), 0, 1).reshape(imstack_long.shape[2], -1)


# optimization of field of view and interaction number
from scipy.optimize import fmin_bfgs

# optimization function
def gbgf(x0):
    # unpack parameters
    sig = x0[0]  # standard deviation of angular gaussian in radians
    intnum = x0[1]  # interaction number
    phi_0 = x0[2]  # central phi in radians
    theta_0 = x0[3]  # central theta in radians

    # calculate angle of each pixel from (phi_0, theta_0)
    radist = 2 * np.arcsin(np.sqrt((np.sin((phi_0 - phi) / 2) ** 2) + np.sin(phi_0) * np.sin(phi) * (np.sin((theta_0 - theta) / 2) ** 2)))

    # calculate gaussian angle weights
    weights = np.exp(- 0.5 * (radist / sig) ** 2)  # gaussian
    weights[np.isnan(phi)] = 0
    weights = weights / np.sum(weights)

    # calculate weighted mean of contact number for each ground points
    w_stack = np.average(imstack_long, weights=weights.ravel(), axis=1)
    # calculate corrcoef with snowfall transmittance over all ground points
    w_corcoef_e = np.corrcoef(hemiList.covariant, np.exp(-intnum * w_stack))[0, 1]

   # return negative for minimization method (we want to maximize corrcoef)
    return -w_corcoef_e


Nfeval = 1
def callbackF(Xi):
    global Nfeval
    print('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}   {5: 3.6f}'.format(Nfeval, Xi[0], Xi[1], Xi[2], Xi[3], gbgf(Xi)))
    Nfeval += 1

print('{0:4s}   {1:9s}   {2:9s}   {3:9s}   {4:9s}   {5:9s}'.format('Iter', ' X1', ' X2', 'X3', 'X4', 'f(X)'))
x0 = np.array([0.11, 21.6, phi[96, 85], theta[96, 85]], dtype=np.double)

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
#
#
# # Here you give the initial parameters for p0 which Python then iterates over
# # to find the best fit
# jj = 90
# ii = 90
#
# popt, pcov = curve_fit(expfunc, w_stack, hemi_var.covariant[hemi_var.training_set.values], p0=(20.0), bounds=(0, np.inf))
#
# plt.scatter(hemi_var.covariant[hemi_var.training_set.values], np.exp(-popt[0] * w_stack), s=2, alpha=.25)

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
abs_max = np.max(np.abs([val_min, val_max]))
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