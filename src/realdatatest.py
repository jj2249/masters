import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pandas as pd
from datahandler import TimeseriesData
from process import *
from particlefilter import RBPF

plt.style.use('ggplot')
### --- importing data --- ###

## - import data from a .csv - ##

data = TimeseriesData(os.pardir+"/resources/data/test_data2.csv")
df_u = data.remove_non_unique(ret=True)
plt.plot(df_u['Date_Time'], df_u['Price'])
plt.xticks([])
plt.show()




### --- RBPF --- ###


## - define particle filter - ##

rbpf = RBPF(mumu=0., beta=0.8, kw=1., kv=1e-6, theta=-15., data=df_u, N=500, gsamps=5_000, epsilon=0.5)
# ## - containers for storing results of rbpf - ##
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

# ## - main loop of rbpf - ##
sm, sv, gm, gv, lml = rbpf.run_filter(ret_history=True)

T = 20

### - plotting results of rbpf - ##
ax1.plot(df_u['Date_Time'][T:], df_u['Price'][T:], label='true')
ax1.plot(rbpf.times[T:], sm[T:])
ax1.fill_between(rbpf.times[T:], (sm-1.96*np.sqrt(sv))[T:], (sm+1.96*np.sqrt(sv))[T:], color='orange', alpha=0.3)
ax2.plot(rbpf.times[T:], gm[T:])
ax2.fill_between(rbpf.times[T:], (gm-1.96*np.sqrt(gv))[T:], (gm+1.96*np.sqrt(gv))[T:], color='orange', alpha=0.3)

ax1.set_xticks([])
ax2.set_xticks([])
fig.legend()
plt.show()
