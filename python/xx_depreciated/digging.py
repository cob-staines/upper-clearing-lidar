import rastools
import numpy as np
import pandas as pd

ras_in = "C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\products\\hs\\19_107\\hs_19_107_res_.04m.tif"
ras = rastools.raster_load(ras_in)

row_map = np.full_like(ras.data, 0).astype(int)
for ii in range(0, ras.rows):
    row_map[ii, :] = ii
col_map = np.full_like(ras.data, 0).astype(int)
for ii in range(0, ras.cols):
    col_map[:, ii] = ii

index_x = np.reshape(row_map, [ras.rows * ras.cols])
index_y = np.reshape(col_map, [ras.rows * ras.cols])
vals = np.reshape(ras.data, [ras.rows * ras.cols])

df = pd.DataFrame({"x": index_x,
                   "y": index_y,
                   "value": vals})

ras.data[np.where(ras.data > 1.25)]

df.loc[df.value > 1.25]
