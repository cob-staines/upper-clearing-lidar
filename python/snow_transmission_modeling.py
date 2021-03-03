import rastools
import numpy as np
import pandas as pd
import tifffile as tif
from scipy.optimize import fmin_bfgs

plot_out_dir = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\graphics\\thesis_graphics\\modeling snow accumulation\\"

batch_dir = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\ray_sampling\\batches\\lrs_hemi_uf_.25m_180px\\outputs\\'
# batch_dir = 'C:\\Users\\jas600\\workzone\\data\\hemigen\\mb_15_1m_pr.15_os10\\outputs\\'

# scaling coefficient converts from expected returns to expected contact number
# scaling_coef = 0.166104  # all rings
scaling_coef = 0.194475  # dropping 5th ring

# load img meta
hemimeta = pd.read_csv(batch_dir + 'rshmetalog.csv')
imsize = hemimeta.img_size_px[0]

# specify which time interval
# interval = "045-050"
interval = "050-052"

# load covariant
count_045 = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_045\\19_045_las_proc\\OUTPUT_FILES\\RAS\\19_045_ground_point_density_r.25m.bil'
count_050 = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_050\\19_050_las_proc\\OUTPUT_FILES\\RAS\\19_050_ground_point_density_r.25m.bil'
count_052 = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_052\\19_052_las_proc\\OUTPUT_FILES\\RAS\\19_052_ground_point_density_r.25m.bil'
count_149 = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\19_149\\19_149_las_proc\\OUTPUT_FILES\\RAS\\19_149_ground_point_density_r.25m.bil'

if interval == "045-050":
    var_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\mb_65\\dSWE\\alin\\interp_2x\\19_045-19_050\\masked\\dswe_alin_19_045-19_050_r.05m_interp2x_masked.tif'
    # var_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\mb_65\\dSWE\\ahpl\\interp_2x\\19_045-19_050\\masked\\dswe_ahpl_19_045-19_050_r.05m_interp2x_masked.tif'
elif interval == "050-052":
    var_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\mb_65\\dSWE\\alin\\interp_2x\\19_050-19_052\\masked\\dswe_alin_19_050-19_052_r.05m_interp2x_masked.tif'
    # var_in = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\mb_65\\dSWE\\ahpl\\interp_2x\\19_050-19_052\\masked\\dswe_ahpl_19_050-19_052_r.05m_interp2x_masked.tif'

ddict = {'count_045': count_045,
         'count_050': count_050,
         'count_052': count_052,
         'count_149': count_149,
         'covariant': var_in}
var = rastools.pd_sample_raster_gdal(ddict, include_nans=True, mode="median")

# # cull outliers
# lb = np.nanquantile(var.covariant, .001)
# ub = np.nanquantile(var.covariant, .999)
# outliers = (var.covariant < lb) | (var.covariant > ub)
# var.loc[outliers, "covariant"] = np.nan

var = var.loc[~np.isnan(var.covariant), :]  # drop nans in covariant

if interval == "045-050":
    var.loc[:, "min_pc"] = np.nanmin((var.count_045, var.count_050, var.count_149), axis=0) * (.25 ** 2)
elif interval == "050-052":
    var.loc[:, "min_pc"] = np.nanmin((var.count_050, var.count_052, var.count_149), axis=0) * (.25 ** 2)

# filter by min point count
var = var.loc[var.min_pc >= 25, :]


# merge with image meta
hemi_var = pd.merge(hemimeta, var, left_on=('x_utm11n', 'y_utm11n'), right_on=('x_coord', 'y_coord'), how='inner')

# load image pixel angle lookup
angle_lookup = pd.read_csv(batch_dir + "phi_theta_lookup.csv")
# build phi image (in radians)
phi = np.full((imsize, imsize), np.nan)
phi[(np.array(angle_lookup.y_index), np.array(angle_lookup.x_index))] = angle_lookup.phi
# build theta image (in radians)
theta = np.full((imsize, imsize), np.nan)
theta[(np.array(angle_lookup.y_index), np.array(angle_lookup.x_index))] = angle_lookup.theta

# limit analysis by phi (in radians)
max_phi = 90 * (np.pi / 180)
# calculate radius range to avoid pixels outside of circle
imrange = np.full((imsize, imsize), False)
imrange[phi <= max_phi] = True

# # filter hemimeta to desired images
# delineate training set (set_param < param_thresh) and test set (set_param >= param thresh)
param_thresh = 0.50
set_param = np.random.random(len(hemi_var))
hemi_var.loc[:, 'training_set'] = set_param < param_thresh
# build hemiList from training_set only
hemiList = hemi_var.loc[hemi_var.training_set, :].reset_index()

# load hemiList images to imstack
imstack = np.full([imsize, imsize, len(hemiList)], np.nan)
for ii in range(0, len(hemiList)):
    imstack[:, :, ii] = tif.imread(batch_dir + hemiList.file_name[ii])[:, :, 0] * scaling_coef
    print(str(ii + 1) + ' of ' + str(len(hemiList)))
#
# # preview of correlation coefficient
# # covar = np.full((imsize, imsize), np.nan)
# corcoef = np.full((imsize, imsize), np.nan)
# corcoef_e = np.full((imsize, imsize), np.nan)
# corcoef_l = np.full((imsize, imsize), np.nan)
# sprank = np.full((imsize, imsize), np.nan)
#
# for ii in range(0, imsize):
#     for jj in range(0, imsize):
#         if imrange[jj, ii]:
#             # covar[jj, ii] = np.cov(hemiList.covariant, imstack[jj, ii, :])[0, 1]
#             corcoef[jj, ii] = np.corrcoef(hemiList.covariant, imstack[jj, ii, :])[0, 1]
#             corcoef_e[jj, ii] = np.corrcoef(hemiList.covariant, np.exp(-21.613288 * imstack[jj, ii, :]))[0, 1]
#             # corcoef_l[jj, ii] = np.corrcoef(hemiList.covariant, np.log(imstack[jj, ii, :]))[0, 1]
#             # sprank[jj, ii] = spearmanr(hemiList.covariant, imstack[jj, ii, :])[0]
#
#     print(ii)
#


# reshape imstack (as imstack_long) for optimization
imstack_long = imstack
imstack_long[np.isnan(imstack_long)] = 0  # careful here..
imstack_long = np.swapaxes(np.swapaxes(imstack_long, 1, 2), 0, 1).reshape(imstack_long.shape[2], -1)


# # optimization function
# def gbgf(x0):
#     # unpack parameters
#     sig = x0[0]  # standard deviation of angular gaussian in radians
#     intnum = x0[1]  # interaction number
#     phi_0 = x0[2]  # central phi in radians
#     theta_0 = x0[3]  # central theta in radians
#     offset = x0[4]
#
#     # calculate angle of each pixel from (phi_0, theta_0)
#     radist = 2 * np.arcsin(np.sqrt((np.sin((phi_0 - phi) / 2) ** 2) + np.sin(phi_0) * np.sin(phi) * (np.sin((theta_0 - theta) / 2) ** 2)))
#
#     # calculate gaussian angle weights
#     weights = np.exp(- 0.5 * (radist / sig) ** 2)  # gaussian
#     weights[np.isnan(phi)] = 0
#     weights = weights / np.sum(weights)
#
#     # calculate weighted mean of contact number for each ground points
#     w_stack = np.average(imstack_long, weights=weights.ravel(), axis=1)
#     # calculate corrcoef with snowfall transmittance over all ground points
#     w_corcoef_e = np.corrcoef(hemiList.covariant + offset, np.exp(-intnum * w_stack))[0, 1]
#
#    # return negative for minimization method (we want to maximize corrcoef)
#     return -w_corcoef_e
#
#
# Nfeval = 1
# def callbackF(Xi):
#     global Nfeval
#     print('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}   {5: 3.6f}   {6: 3.6f}'.format(Nfeval, Xi[0], Xi[1], Xi[2], Xi[3], Xi[4], gbgf(Xi)))
#     Nfeval += 1
#
# print('{0:4s}   {1:9s}   {2:9s}   {3:9s}   {4:9s}   {5:9s}   {6:9s}'.format('Iter', ' X1', ' X2', 'X3', 'X4', 'X5', 'f(X)'))
# x0 = np.array([0.11, 21.6, phi[96, 85], theta[96, 85], 0], dtype=np.double)
# x0 = np.array([9.73075682e-02, 1.53251521e+01, 1.20260398e-01, 5.40964306e+00, -1.21090673e-02])  # 19_045-19_050
# [xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflg] = \
#     fmin_bfgs(gbgf, x0, callback=callbackF, maxiter=5, full_output=True, retall=False)
# # xopt = np.array([0.1296497, 21.57953188, 96.95887751, 86.24391083])  # gaussian optimization,
# # fopt = -0.4744932

from scipy.stats import vonmises
from scipy.special import i0

def dwst(p0):
    # unpack parameters
    sig = p0[0]  # standard deviation of angular gaussian in radians
    intnum = p0[1]  # interaction number
    phi_0 = p0[2]  # central phi in radians
    theta_0 = p0[3]  # central theta in radians
    mm = p0[4]
    # bb = p0[5]

    # calculate angle of each pixel from (phi_0, theta_0)
    radist = np.arccos(np.cos(phi_0) * np.cos(phi) + np.sin(phi_0) * np.sin(phi) * np.cos(theta_0 - theta))

    # radist = 2 * np.arcsin(np.sqrt((np.cos((phi_0 - phi) / 2) ** 2) + np.sin(phi_0) * np.sin(phi) * (np.sin((theta_0 - theta) / 2) ** 2)))
    # calculate gaussian angle weights
    weights = np.exp(- 0.5 * (radist / sig) ** 2)  # gaussian

    # # calculate gaussian/vonmises weights
    # weights = np.exp(- 0.5 * (phi_0 - phi / sig) ** 2) * np.exp(kk * np.cos(theta - theta_0)) / i0(kk)  # gaussian * von mises

    weights[np.isnan(phi)] = 0
    weights = weights / np.sum(weights)

    # calculate weighted mean of contact number for each ground points
    w_stack = np.average(imstack_long, weights=weights.ravel(), axis=1)

    # model snow accumulation with snowfall transmittance over all ground points
    snowacc = mm * np.exp(-intnum * w_stack)
    # snowacc = mm * np.exp(-intnum * w_stack) + bb

    dswe = hemiList.covariant

    # calculate sum of square residuals
    ssres = np.sum((dswe - snowacc) ** 2)

    return ssres

def rsq(p0):
    ssres = dwst(p0)

    dswe = hemiList.covariant

    sstot = np.sum((dswe - np.mean(dswe)) ** 2)
    return 1 - ssres / sstot

Nfeval = 1
def callbackF(Xi):
    global Nfeval
    print('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}   {5: 3.6f}   {6: 3.6f}   {7: 3.6f}'.format(Nfeval, Xi[0], Xi[1], Xi[2], Xi[3], Xi[4], dwst(Xi), rsq(Xi)))
    # print('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}   {5: 3.6f}   {6: 3.6f}   {7: 3.6f}   {8: 3.6f}'.format(Nfeval, Xi[0], Xi[1], Xi[2], Xi[3], Xi[4], Xi[5], dwst(Xi), rsq(Xi)))
    Nfeval += 1

print('{0:4s}   {1:9s}   {2:9s}   {3:9s}   {4:9s}   {5:9s}   {6:9s}   {7:9s}'.format('Iter', ' sig', ' intnum', 'phi', 'theta', 'mm','f(X)', 'R2'))
# print('{0:4s}   {1:9s}   {2:9s}   {3:9s}   {4:9s}   {5:9s}   {6:9s}   {7:9s}   {8:9s}'.format('Iter', ' sig', ' intnum', 'phi', 'theta', 'mm', 'bb', 'f(X)', 'R2'))


if interval == "045-050":
    # p0 = np.array([0.11133662, 1.18626547, 0.09616933, 2.49615144, 8.15912542])  # 19_045-19_050, min_ct >= 10, alin, no bb, r2 = 0.172494
    # p0 = np.array([0.12009855, 1.72308774, 0.09704997, 2.50291223, 10.0493321])  # 19_045-19_050, min_ct >= 10, ahpl, no bb, r2 = 0.258417
    # p0 = np.array([0.11779021, 0.88873142, 0.09885492, 2.48813593, 9.52537669, -1.45475159])  # 19_045-19_050, alin, min_ct >= 10, alin, r2 = 0.173103
    # p0 = np.array([0.13003656, 1.25413136, 0.09995546, 2.49116541, 11.42591721, -1.56140183])  # 19_045-19_050, ahpl, min_ct >= 10, ahpl, r2 = 0.260354

    p0 = np.array([0.11868072, 1.09587946, 0.10320082, 2.51346561, 8.24466212])  # 19_045-19_050, min_ct >= 25, alin, no bb, 50% of data, r2 = 0.246279
    p0 = np.array([0.12311107, 1.53782685, 0.09861554, 2.49519406, 9.51433871])  # 19_045-19_050, min_ct >= 25, ahpl, no bb, 50% of data, r2 = 0.322026

elif interval == "050-052":
    # p0 = np.array([0.16484606, 2.26669136, 0.18210887, 2.52394167, 18.89760106])  # 19_050-19_052 (dropping min_ct < 10), alin, no bb, r2 = 0.265145  -- these look bad... transmission too low
    # p0 = np.array([0.1713302, 1.16210526, 0.18899996, 2.54200686, 16.80013182])  # 19_050-19_052, min_ct >= 10, ahpl, no bb, r2 = 0.209451 -- still somewhat of a blob. Could it be that outliers are responsible?
    # p0 = np.array([0.1355174, 3.34151497, 0.1542143, 2.55194135, 17.27915456, 1.68683719])  # 19_050-19_052 (dropping min_ct < 10), alin, r2 = 0.272531 -- these look bad... transmission too low
    # p0 = np.array([0.13587388, 2.42821223, 0.16443314, 2.62978447, 13.41904737, 4.3388157])  # 19_050-19_052, min_ct >= 10, ahpl, r2 = 0.217985 -- same problem, absorption factor too high, transmission too low.


    p0 = np.array([ 0.15603763,  2.16304588,  0.15021845,  2.66995801, 18.30307134])  # 19_050-19_052, min_ct >= 25, alin, no bb, 50% of data, r2 = 0.409706
    p0 = np.array([0.16483349, 1.08482362, 0.18124984, 2.60465732, 16.32643726])  # 19_050-19_052, min_ct >= 25, ahpl, no bb, 50% of data, r2 = 0.286698

# run optimization
[popt, fopt, gopt, Bopt, func_calls, grad_calls, warnflg] = \
    fmin_bfgs(dwst, p0, callback=callbackF, maxiter=50, full_output=True, retall=False)

u_out = (180 / np.pi, 1, 180 / np.pi, 180/np.pi, 1)



#
# sig_out = xopt[0] * 180 / np.pi
# intnum_out = 1/xopt[1]
# phi_out = np.sqrt((xopt[2] - (imsize - 1)/2) ** 2 + (xopt[3] - (imsize - 1)/2) ** 2)
# theta_out = np.arctan2(-(xopt[3] - (imsize - 1)/2), -(xopt[2] - (imsize - 1)/2)) * 180 / np.pi
#



####### Visualization
import matplotlib
matplotlib.use('Qt5Agg')
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

# p0 = popt.copy()

sig = p0[0]  # standard deviation of angular gaussian in radians
intnum = p0[1]  # interaction number
phi_0 = p0[2]  # central phi in radians
theta_0 = p0[3]  # central theta in radians
mm = p0[4]
# bb = p0[5]
bb = 0

# calculate angle of each pixel from (phi_0, theta_0)
radist = np.arccos(np.cos(phi_0) * np.cos(phi) + np.sin(phi_0) * np.sin(phi) * np.cos(theta_0 - theta))

# radist = 2 * np.arcsin(np.sqrt((np.cos((phi_0 - phi) / 2) ** 2) + np.sin(phi_0) * np.sin(phi) * (np.sin((theta_0 - theta) / 2) ** 2)))
# calculate gaussian angle weights
weights = np.exp(- 0.5 * (radist / sig) ** 2)  # gaussian

# # calculate gaussian/vonmises weights
# weights = np.exp(- 0.5 * (phi_0 - phi / sig) ** 2) * np.exp(kk * np.cos(theta - theta_0)) / i0(kk)  # gaussian * von mises

weights[np.isnan(phi)] = 0
weights = weights / np.sum(weights)

# calculate weighted mean of contact number for each ground points
w_stack = np.average(imstack_long, weights=weights.ravel(), axis=1)

# model snow accumulation with snowfall transmittance over all ground points
# snowacc = mm * np.exp(-intnum * w_stack) + bb
transmittance = np.exp(-intnum * w_stack)

fig = plt.figure()
fig.subplots_adjust(top=0.90, bottom=0.12, left=0.12)
ax1 = fig.add_subplot(111)
if interval == "045-050":
    ax1.set_title('$\Delta$SWE vs. directionally weighted snowfall transmission $T^{*}_{(x, y)}$\n Upper Forest, 14-19 Feb. 2019, 25cm resolution')
elif interval == "050-052":
    ax1.set_title('$\Delta$SWE vs. directionally weighted snowfall transmission $T^{*}_{(x, y)}$\n Upper Forest, 19-21 Feb. 2019, 25cm resolution')
ax1.set_xlabel("$\Delta$SWE [mm]")
ax1.set_ylabel("$T^{*}_{(x, y)}$ [-]")
plt.scatter(hemiList.covariant, transmittance, s=2, alpha=.25)
mm_mod = np.array([np.nanmin(transmittance), np.nanmax(transmittance)])
plt.plot(mm_mod * mm + bb, mm_mod, c='Black', linestyle='dashed', linewidth=1.5)
plt.ylim(0, 1)
if interval == "045-050":
    fig.savefig(plot_out_dir + "dSWE vs DWST 045-050.png")
elif interval == "050-052":
    fig.savefig(plot_out_dir + "dSWE vs DWST 050-052.png")

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
w_data = pd.DataFrame(columns={"sig", "r2"})
ii = 0
px = p0.copy()
for vv in v_list:

    px[0] = vv
    r2 = rsq(px)

    new_row = {"sig": vv,
               "r2": r2}
    w_data = w_data.append(new_row, ignore_index=True, verify_integrity=True)

    ii += 1
    print(ii)

fig = plt.figure()
fig.subplots_adjust(top=0.90, bottom=0.15, left=0.15)
ax1 = fig.add_subplot(111)
if interval == "045-050":
    ax1.set_title('Optimization of Gaussian weight function width $\sigma$\nUpper Forest, 14-19 Feb. 2019, 25cm resolution')
elif interval == "050-052":
    ax1.set_title('Optimization of Gaussian weight function width $\sigma$\nUpper Forest, 19-21 Feb. 2019, 25cm resolution')
ax1.set_ylabel("$R^2$ for $\Delta$SWE vs. modeled snow accumulation")
ax1.set_xlabel("Standard deviation of Gausian weight function $\sigma$ [$^{\circ}$]")
plt.plot(w_data.sig * 180 / np.pi, w_data.r2)
if interval == "045-050":
    fig.savefig(plot_out_dir + "optimization of gaussian weight function width sigma 045-050.png")
elif interval == "050-052":
    fig.savefig(plot_out_dir + "optimization of gaussian weight function width sigma 050-052.png")

#####
# interaction scalar
# sample optimization topography for mu*
v_list = np.linspace(.1, 5, 100)  # mu*
w_data = pd.DataFrame(columns={"mu", "r2"})
ii = 0
px = p0.copy()
for vv in v_list:

    px[1] = vv
    r2 = rsq(px)

    new_row = {"mu": vv,
               "r2": r2}
    w_data = w_data.append(new_row, ignore_index=True, verify_integrity=True)

    ii += 1
    print(ii)

fig = plt.figure()
fig.subplots_adjust(top=0.90, bottom=0.15, left=0.15)
ax1 = fig.add_subplot(111)
if interval == "045-050":
    ax1.set_title('Optimization of snowfall absorbtion coefficient $\mu^*$\nUpper Forest, 14-19 Feb. 2019, 25cm resolution')
elif interval == "050-052":
    ax1.set_title('Optimization of snowfall absorbtion coefficient $\mu^*$\nUpper Forest, 19-21 Feb. 2019, 25cm resolution')
ax1.set_ylabel("$R^2$ for $\Delta$SWE vs. modeled snow accumulation")
ax1.set_xlabel("snowfall absorbtion coefficient $\mu^*$ [-]")
plt.plot(w_data.mu, w_data.r2)
plt.ylim(-1, )
if interval == "045-050":
    fig.savefig(plot_out_dir + "optimization of snow absorption coefficient mu 045-050.png")
elif interval == "050-052":
    fig.savefig(plot_out_dir + "optimization of snow absorption coefficient mu 050-052.png")



################################################
## plot hemispherical footprint

intnum = p0[1]

# rerun corcoef_e with optimized transmission scalar
corcoef_e = np.full((imsize, imsize), np.nan)
for ii in range(0, imsize):
    for jj in range(0, imsize):
        if imrange[jj, ii]:
            corcoef_e[jj, ii] = np.corrcoef(hemiList.covariant, np.exp(-intnum * imstack[jj, ii, :]))[0, 1]

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

# plot with axes
fig = plt.figure(figsize=(7, 7))
if interval == "045-050":
    fig.suptitle("Correlation of $\Delta$SWE with snowfall transmission over hemisphere\nUpward-looking, Upper Forest, 14-19 Feb 2019, 25cm resolution")
elif interval == "050-052":
    fig.suptitle("Correlation of $\Delta$SWE with snowfall transmission over hemisphere\nUpward-looking, Upper Forest, 19-21 Feb 2019, 25cm resolution")
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
ax.set_thetagrids(np.linspace(0, 360, 4, endpoint=False), labels=['N\n  0$^\circ$', 'W\n  270$^\circ$', 'S\n  180$^\circ$', 'E\n  90$^\circ$'])

# add colorbar
fig.subplots_adjust(top=0.95, left=0.1, right=0.75, bottom=0.05)
cbar_ax = fig.add_axes([0.85, 0.20, 0.03, 0.6])
fig.colorbar(im, cax=cbar_ax)
cbar_ax.set_ylabel("Pearson's correlation coefficient")

if interval == "045-050":
    fig.savefig(plot_out_dir + "footprint corcoef dswe with snowfall transmission 045-050.png")
elif interval == "050-052":
    fig.savefig(plot_out_dir + "footprint corcoef dswe with snowfall transmission 050-052.png")


# plot with contours
# plot with axes
fig = plt.figure(figsize=(7, 7))
if interval == "045-050":
    fig.suptitle("Correlation of $\Delta$SWE with snowfall transmission over hemisphere\nUpward-looking, Upper Forest, 14-19 Feb 2019, 25cm resolution")
elif interval == "050-052":
    fig.suptitle("Correlation of $\Delta$SWE with snowfall transmission over hemisphere\nUpward-looking, Upper Forest, 19-21 Feb 2019, 25cm resolution")
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
ax.set_thetagrids(np.linspace(0, 360, 4, endpoint=False), labels=['N\n  0$^\circ$', 'W\n  270$^\circ$', 'S\n  180$^\circ$', 'E\n  90$^\circ$'])

# add colorbar
fig.subplots_adjust(top=0.95, left=0.1, right=0.75, bottom=0.05)
cbar_ax = fig.add_axes([0.85, 0.20, 0.03, 0.6])
fig.colorbar(im, cax=cbar_ax)
cbar_ax.set_ylabel("Pearson's correlation coefficient")

# contours
ax.set_rgrids([])  # no rgrids
ax.grid(False)  # no grid
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
matplotlib.rcParams["lines.linewidth"] = 1
CS = ax0.contour(set, np.linspace(-.4, .4, 9), colors="k")
plt.clabel(CS, inline=1, fontsize=8)

if interval == "045-050":
    fig.savefig(plot_out_dir + "footprint corcoef dswe with snowfall transmission 045-050 contours.png")
elif interval == "050-052":
    fig.savefig(plot_out_dir + "footprint corcoef dswe with snowfall transmission 050-052 contours.png")

#
# ###
# # plot radial distribution of corcoef_e
# sig = xopt[0]
# cpy = xopt[2]
# cpx = xopt[3]
# yid, xid = np.indices((imsize, imsize))
# radist = np.sqrt((yid - cpy) ** 2 + (xid - cpx) ** 2) * np.pi / 180
#
# radcor = pd.DataFrame({"phi": np.rint(np.ravel(radist) * 180 / np.pi).astype(int),
#                        "corcoef_e": np.ravel(corcoef_e)})
#
# radmean = radcor.groupby('phi').mean().reset_index(drop=False)
# radmean = radmean.assign(gaus=np.exp(- 0.5 * (radmean.phi * np.pi / (sig * 180)) ** 2))
#
# plt.plot(radmean.phi, radmean.corcoef_e)
# plt.plot(radmean.phi, radmean.gaus * np.nanmax(radmean.corcoef_e))
# plt.plot(radmean.phi, radmean.corcoef_e - radmean.gaus * np.nanmax(radmean.corcoef_e))
#
# ###
#
# maxcoords = np.where(corcoef_e == np.nanmax(corcoef_e))
# jj = maxcoords[0][0]
# ii = maxcoords[1][0]
#
# plt.scatter(hemiList.covariant, np.exp(-xopt[1] * imstack[jj, ii, :]), alpha=0.1, s=5)
#
# immid = ((imsize - 1)/2).astype(int)
# jj = immid
# ii = immid
# plt.scatter(hemiList.covariant, np.exp(-xopt[1] * imstack[jj, ii, :]), alpha=0.1, s=5)
# # cumulative weights... this is all messed
#
# plt.imshow(sprank, cmap=plt.get_cmap('Purples_r'))
# plt.colorbar()
#

# angle_lookup_covar.sqr_covar_weighted
# angle_lookup_covar.sort_values(phi)
# peace = angle_lookup_covar.loc[~np.isnan(angle_lookup_covar.covar), :]
# peace.loc[:, 'sqr_covar_weight_cumsum'] = np.cumsum(peace.sort_values('phi').sqr_covar_weight.values).copy()
#
# plt.scatter(peace.phi, peace.sqr_covar_weight_cumsum)