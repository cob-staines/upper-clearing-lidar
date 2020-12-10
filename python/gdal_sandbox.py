# from osgeo import gdal, gdalconst
# import numpy as np
#
# # Source
# src_filename = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\qc_test\\19_149_dem_r.25m_count_crop1.tif'
# src = gdal.Open(src_filename, gdalconst.GA_ReadOnly)
# src_proj = src.GetProjection()
# src_geotrans = src.GetGeoTransform()
#
# # We want a section of source that matches this:
# match_filename = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\qc_test\\19_149_dem_r.25m_count_crop2.tif'
# match_ds = gdal.Open(match_filename, gdalconst.GA_ReadOnly)
# match_proj = match_ds.GetProjection()
# match_geotrans = match_ds.GetGeoTransform()
# wide = match_ds.RasterXSize
# high = match_ds.RasterYSize
#
# # # Output / destination
# # dst_filename = 'C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\qc_test\\19_149_dem_r.25m_count_crop2_reprj.tif'
# # dst = gdal.GetDriverByName('GTiff').Create(dst_filename, wide, high, 1, gdalconst.GDT_Float32)
# # dst.SetGeoTransform(match_geotrans)
# # dst.SetProjection(match_proj)
#
# mem_drv = gdal.GetDriverByName('MEM')
# dest = mem_drv.Create('', wide, high, 1, gdal.GDT_Float32)
# dest.GetRasterBand(1).WriteArray(np.full((high, wide), np.nan), 0, 0)
# # Set the geotransform
# dest.SetGeoTransform(match_geotrans)
# dest.SetProjection(match_proj)
# # Perform the projection/resampling
# res = gdal.ReprojectImage(src, dest, src_proj, match_proj, gdal.GRA_Bilinear)
#
# ofnp = myarray = np.array(dest.GetRasterBand(1).ReadAsArray())

# # Do the work
# gdal.ReprojectImage(src, dst, src_proj, match_proj, gdalconst.GRA_Bilinear)
# del dst # Flush


# import multiprocessing as mp
#
# mp.cpu_count()
#
# def spawn(num):
#     print('Spawn # {}'.format(num))
#
# if __name__ == '__main__':
#     for i in range(5):
#         p = mp.Process(target=spawn, args=(i,))
#         p.start()

# import time
# from multiprocessing import Pool
#
#
# def square(x):
#     print(f"start process:{x}")
#     square = x * x
#     print(f"square {x}:{square}")
#     time.sleep(1)
#     if square % 2 == 0:
#         time.sleep(.5)
#         print(f"{square} is even")
#     print(f"end process:{x}")
#
#
# if __name__ == "__main__":
#     starttime = time.time()
#     pool = Pool()
#     pool.map(square, range(0, 10))
#     pool.close()
#     endtime = time.time()
#     print(f"Time taken {endtime-starttime} seconds")

from multiprocessing import Process
import sharedmem
import numpy

def do_work(data, start):
    print(data[start])

def split_work(num):
    n = 20
    width = int(n/num)
    shared = sharedmem.empty(n)
    shared[:] = numpy.random.rand(1, n)[0]
    print("values are %s" % shared)

    processes = [Process(target=do_work, args=(shared, i*width), daemon=False) for i in range(0, num)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    print("values are %s" % shared)
    print("type is %s" % type(shared[0]))

if __name__ == '__main__':
    split_work(4)

# from cachetools import cached, LRUCache
# cache = LRUCache(maxsize=2)
# import time
#
# @cached(cache)
# def sleep_squared(s):
#     time.sleep(s)
#     return s ** 2
#
# sleep_squared(2)
#
# ##
# import theano
# print(theano.config.device)

import pandas as pd
import numpy as np

# load data
data_in ='C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\analysis\\mb_15_merged_.25m_canopy_19_149.csv'
data = pd.read_csv(data_in)

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

data = data.loc[data.uf.values == 1, :]

plt.scatter(data.loc[:, "dswe_19_045-19_052"], np.log(data.lai_s_cc), s=1, alpha=.25)
plt.scatter(data.loc[:, "dswe_19_045-19_052"], np.log(data.er_p0_mean * 0.19546), s=1, alpha=.25)



fig = plt.figure(figsize=(5, 10))
ax1 = fig.add_subplot(211)


ax1.set_ylabel('ln(Contact Number Weighted)')
hmm = plt.scatter(data.loc[:, "dswe_19_045-19_052"], np.log(data.cn_weighted), s=1, alpha=.25)
plt.xlim(-20, 60)

ax2 = fig.add_subplot(212)
ax2.set_ylabel('DCE')
hmm = plt.scatter(data.loc[:, "dswe_19_045-19_052"], data.loc[:, "dce"], s=1, alpha=.25)
plt.xlim(-20, 60)

ax2.set_xlabel('$\Delta$ SWE')
plt.show()

fig = plt.figure(figsize=(5, 5))
ax3 = fig.add_subplot(111)
ax3.set_xlabel('DCE')
ax3.set_ylabel('ln(Contact Number Weighted)')
plt.scatter(data.loc[:, "dce"], np.log(data.cn_weighted), s=1, alpha=.25, c='green')




fig, ax = plt.subplot()
hmm = ax.scatter(data.loc[:, "dswe_19_045-19_052"], np.log(data.cn_weighted), s=1, alpha=.25)
plt.xlim(-20, 60)
plt.set_xlabel("$\\Delta SWE")
plt.set_ylabel("ln(Weighted Contact Number)")

plt.subplot(2, 2, 2)
plt.scatter(data.loc[:, "dswe_19_045-19_052"], data.loc[:, "dce"], s=1, alpha=.25)
plt.xlim(-20, 60)

plt.subplot(2,2,3)
plt.scatter(data.loc[:, "dce"], np.log(data.cn_weighted), s=1, alpha=.25)

###

plt.scatter(data.loc[:, "chm"], data.er_p0_mean, s=1, alpha=.25)
plt.scatter(data.loc[:, "dswe_19_045-19_052"], np.log(data.loc[:, "chm"]), s=1, alpha=.25)
plt.scatter(data.loc[:, "dswe_19_045-19_052"], np.log(data.loc[:, "er_p0_mean"]), s=1, alpha=.25)
plt.scatter(data.loc[:, "dswe_19_045-19_052"], data.loc[:, "dce"], s=1, alpha=.25)
plt.scatter(data.loc[:, "dswe_19_045-19_052"], np.log(data.cn_weighted), s=1, alpha=.05)