import numpy as np
import pandas as pd
from windrose import WindroseAxes
from matplotlib import pyplot as plt
import matplotlib.cm as cm


def load_met_data(file_in, daterange=None):
    data_head = pd.read_csv(file_in, header=0, skiprows=1, na_values=[-9999.0, 'NA']).columns
    data_raw = pd.read_csv(file_in, header=None, skiprows=4, na_values=[-9999.0, 'NA'])
    data_raw.columns = data_head
    data_raw.TIMESTAMP = pd.to_datetime(data_raw.TIMESTAMP)

    if daterange is not None:
        inrange = (data_raw.TIMESTAMP >= daterange[0]) & (data_raw.TIMESTAMP <= daterange[1])
        data = data_raw.loc[inrange, :]

    return data


def contour_wind_plot(data, binstep=1.0, max_ws=None, flip=False):
    if max_ws is None:
        max_ws = np.max(data.loc[:, 'Wind Speed'])
    ax = WindroseAxes.from_ax()

    if flip:
        wd = 360 -data.loc[:, 'Wind Direction']
        # wd_labels = ["W\n 270$^\circ$", "N-W\n 315$^\circ$", "N\n  0$^\circ$", "N-E\n  45$^\circ$", "E\n  90$^\circ$",
        #              "S-E\n 135$^\circ$", "S\n 180$^\circ$", "S-W\n 225$^\circ$"]
        wd_labels = ["W\n 270$^\circ$", "", "N\n  0$^\circ$", "N-E\n  45$^\circ$", "E\n  90$^\circ$",
                     "S-E\n 135$^\circ$", "S\n 180$^\circ$", "S-W\n 225$^\circ$"]
    else:
        wd = data.loc[:, 'Wind Direction']
        # wd_labels = ["E\n  90$^\circ$", "N-E\n  45$^\circ$", "N\n  0$^\circ$", "N-W\n 315$^\circ$", "W\n 270$^\circ$",
        #              "S-W\n 225$^\circ$", "S\n 180$^\circ$", "S-E\n 135$^\circ$"]
        wd_labels = ["E\n  90$^\circ$", "", "N\n  0$^\circ$", "N-W\n 315$^\circ$", "W\n 270$^\circ$",
                     "S-W\n 225$^\circ$", "S\n 180$^\circ$", "S-E\n 135$^\circ$"]
    ws = data.loc[:, 'Wind Speed']
    # ax.contourf(wd, ws, bins=np.arange(0, max_ws + 1 * binstep, binstep), cmap=cm.viridis, normed=True)
    ax.bar(wd, ws, bins=np.arange(0, max_ws + 1 * binstep, binstep), normed=True, opening=1, edgecolor='white', cmap=cm.viridis)
    ax.set_legend(title="Wind speed [m/s]", loc='upper right', decimal_places=2)


    ax.set_thetagrids(np.linspace(0, 360, 8, endpoint=False), labels=wd_labels)

    # ax.theta_labels = ["E", "N-E", "N", "N-W", "W", "S-W", "S", "S-E"]

    return ax

plot_out_dir = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\graphics\\thesis_graphics\\met station\\"

daterange_1 = [np.datetime64('2019-02-14T12:00'), np.datetime64('2019-02-19T12:00')]
daterange_2 = [np.datetime64('2019-02-19T12:00'), np.datetime64('2019-02-21T12:00')]

# uc_in = "C:/Users/Cob/index/educational/usask/research/masters/data/met/2016_19_QC_data_merge/Upper_Clearing_15min_2016_19_slim_merge.txt"
# uc_1 = load_met_data(uc_in, daterange_1)
# uc_2 = load_met_data(uc_in, daterange_2)
#
uf_in = "C:/Users/Cob/index/educational/usask/research/masters/data/met/2016_19_QC_data_merge/Upper_Forest_15min_2016_19_slim_merge.txt"
uf_1 = load_met_data(uf_in, daterange_1)
uf_2 = load_met_data(uf_in, daterange_2)

ut_in = "C:/Users/Cob/index/educational/usask/research/masters/data/met/2016_19_QC_data_merge/Upper_Clearing_Tower_15min_2016_19_slim_merge.txt"
ut_1 = load_met_data(ut_in, daterange_1)
ut_2 = load_met_data(ut_in, daterange_2)

max_ws = np.max([np.max(ut_1.loc[:, 'Wind Speed'].values), np.max(ut_2.loc[:, 'Wind Speed'].values)])
bs = .25
fig = plt.figure()
# ut_1_plot = contour_wind_plot(ut_1, binstep=.25, max_ws=max_ws)
ut_1_plot = contour_wind_plot(ut_1, binstep=bs, max_ws=max_ws, flip=True)
wind_dir = ut_1_plot._info["dir"]
wind_spd = ut_1_plot._info["bins"]
ut_1_table = ut_1_plot._info["table"]

# contour_wind_plot(uc_1, binstep=.25, max_ws=max_ws)
# contour_wind_plot(uf_1, binstep=.25, max_ws=max_ws)
# ut_2_plot = contour_wind_plot(ut_2, binstep=.25, max_ws=max_ws)
ut_2_plot = contour_wind_plot(ut_2, binstep=bs, max_ws=max_ws, flip=True)
ut_2_table = ut_2_plot._info["table"]
# contour_wind_plot(uc_2, binstep=.25, max_ws=max_ws)
# contour_wind_plot(uf_2, binstep=.25, max_ws=max_ws)


# descriptive stats
wd_1 = np.sum(ut_1_table, axis=0)
ws_1 = np.sum(ut_1_table, axis=1)
ws_cs_1 = np.cumsum(ws_1)
ws_csr_1 = 100 - ws_cs_1

wd_2 = np.sum(ut_2_table, axis=0)
ws_2 = np.sum(ut_2_table, axis=1)
ws_cs_2 = np.cumsum(ws_2)
ws_csr_2 = 100 - ws_cs_2

plt.plot(np.array(wind_spd[:-1]) + bs / 2, ws_1)
plt.plot(np.array(wind_spd[:-1]) + bs / 2, ws_2)

plt.plot(np.array(wind_spd[:-1]) + bs / 2, ws_cs_1)
plt.plot(np.array(wind_spd[:-1]) + bs / 2, ws_cs_2)

plt.plot(np.array(wind_spd[:-1]) + bs / 2, ws_csr_1)
plt.plot(np.array(wind_spd[:-1]) + bs / 2, ws_csr_2)

np.quantile(ut_1.loc[:, 'Wind Speed'], q=.5)
np.quantile(ut_2.loc[:, 'Wind Speed'], q=.5)

np.quantile(ut_1.loc[:, 'Wind Speed'], q=.75)
np.quantile(ut_2.loc[:, 'Wind Speed'], q=.75)

np.sum(ut_1.loc[:, 'Wind Speed'] > .25) / ut_1.__len__()
np.sum(ut_2.loc[:, 'Wind Speed'] > .25) / ut_2.__len__()

np.sum(ut_1.loc[:, 'Wind Speed'] > .5) / ut_1.__len__()
np.sum(ut_2.loc[:, 'Wind Speed'] > .5) / ut_2.__len__()

np.sum(ut_1.loc[:, 'Wind Speed'] > .75) / ut_1.__len__()
np.sum(ut_2.loc[:, 'Wind Speed'] > .75) / ut_2.__len__()

np.sum(ut_1.loc[:, 'Wind Speed'] > 1) / ut_1.__len__()
np.sum(ut_2.loc[:, 'Wind Speed'] > 1) / ut_2.__len__()

np.sum(ut_1.loc[:, 'Wind Speed'] > 1.25) / ut_1.__len__()
np.sum(ut_2.loc[:, 'Wind Speed'] > 1.25) / ut_2.__len__()

np.sum(ut_1.loc[:, 'Wind Speed'] > 1.5) / ut_1.__len__()
np.sum(ut_2.loc[:, 'Wind Speed'] > 1.5) / ut_2.__len__()

np.max(ut_1.loc[:, 'Wind Speed'])
np.max(ut_2.loc[:, 'Wind Speed'])

np.max(ut_1.loc[:, 'Air Temperature'])
np.min(ut_1.loc[:, 'Air Temperature'])
np.mean(ut_1.loc[:, 'Air Temperature'])
np.median(ut_1.loc[:, 'Air Temperature'])


np.max(ut_2.loc[:, 'Air Temperature'])
np.min(ut_2.loc[:, 'Air Temperature'])
np.mean(ut_2.loc[:, 'Air Temperature'])
np.median(ut_2.loc[:, 'Air Temperature'])

np.max(uf_1.loc[:, "Wind Speed"])
np.max(uf_2.loc[:, "Wind Speed"])