import numpy as np
import matplotlib.pyplot as plt
import os

from datahandler import TimeseriesData
from process import *


### --- Forward Simulation --- ###
vg = VarianceGammaProcess(10., 0.1, 0., 1.)
vg.generate()

lss = LangevinModel(vg, -5.)
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
lss.plot_timeseries(fig.axes)
plt.show()

### --- importing data --- ### 
# data = TimeseriesData(os.pardir+"/resources/data/test_data.csv")
# df_u = data.remove_non_unique(ret=True)
# plt.plot(df_u['Time'], df_u['Price'])
# plt.xticks([])
# plt.show()


