
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors

df_in ='C:\\Users\\Cob\\index\\educational\\usask\\research\\masters\\data\\lidar\\analysis\\uf_merged_ajli_.10m_canopy_19_149.csv'
df = pd.read_csv(df_in)
df.loc[:, "cn"] = df.er_p0_mean * 0.19447

sns.scatterplot(data=df, x="dswe_19_045-19_050", y="cn")

fig, ax = plt.subplots(figsize=(8, 5.7))
ax.scatter(df["dswe_19_045-19_050"], df["cn"], alpha=0.1, s=4)


(np.array([8, 5.7]) * 73).astype(int)

x_dat = df["dswe_19_045-19_050"]
y_dat = df["cn"]
plt.hist2d(x_dat, y_dat, range=[[np.nanquantile(x_dat, .001), np.nanquantile(x_dat, .999)], [np.nanmin(y_dat), np.nanquantile(y_dat, .999)]], bins=(np.array([8, 5.7]) * 20).astype(int), norm=colors.LogNorm(), cmap="Blues")
