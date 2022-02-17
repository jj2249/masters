import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pandas as pd
from datahandler import TimeseriesData
from process import *
from particlefilter import RBPF


### --- importing data --- ###

## - import data from a .csv - ##

data = TimeseriesData(os.pardir+"/resources/data/test_data2.csv")
df_u = data.remove_non_unique(ret=True)
plt.plot(df_u['Date_Time'], df_u['Price'])
plt.xticks([])
plt.show()




### --- RBPF --- ###


## - define particle filter - ##

rbpf = RBPF(mumu=0., sigmasq=.0001, beta=0.2, kw=1e6, kv=.05, theta=-0.8, data=df_u, N=1_000, gsamps=10_000, epsilon=0.5)
# ## - containers for storing results of rbpf - ##
fig = plt.figure()
ax1 = fig.add_subplot(111)
# ax2 = fig.add_subplot(212)

# ## - main loop of rbpf - ##
sm, sv, gm, gv, lml = rbpf.run_filter(ret_history=True)

### - plotting results of rbpf - ##
ax1.plot(df_u['Date_Time'], df_u['Price'], label='true')
ax1.plot(rbpf.times+rbpf.initial_time, sm+rbpf.initial_price)
ax1.fill_between(rbpf.times+rbpf.initial_time, sm+rbpf.initial_price-1.96*np.sqrt(sv), sm+rbpf.initial_price+1.96*np.sqrt(sv), color='orange', alpha=0.3)


ax1.set_xticks([])
# ax2.set_xticks([])
fig.legend()
plt.show()
