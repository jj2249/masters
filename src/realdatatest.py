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

tobj = TimeseriesData(os.pardir+"/resources/data/oildata.csv")
plt.plot(tobj.df['Telapsed'], tobj.df['Price'])
# plt.xticks([])
plt.xlabel('Elapsed time (minutes)')
plt.ylabel('Price')

plt.show()




### --- RBPF --- ###


## - define particle filter - ##

# rbpf = RBPF(mumu=0., beta=0.8, kw=1., kv=1e-6, theta=-15., data=df_u, N=500, gsamps=5_000, epsilon=0.5)
rbpf = RBPF(mux=110., mumu=0., beta=5., kw=2., kv=1e-2, kmu=1e-2, rho=1e-5, eta=1e-5, theta=-5., data=tobj.df, N=500, gsamps=100, epsilon=0.5)

# ## - containers for storing results of rbpf - ##
fig = plt.figure()
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)

# ## - main loop of rbpf - ##
sm, sv, gm, gv, mm, mv, lml = rbpf.run_filter(ret_history=True, tpred=15.)

T = 0

### - plotting results of rbpf - ##
ax1.plot(tobj.df['Telapsed'][T:], tobj.df['Price'][T:], label='true')
ax1.plot(rbpf.times[T:], sm[T:])
ax1.fill_between(rbpf.times[T:], (sm-1.96*np.sqrt(sv))[T:], (sm+1.96*np.sqrt(sv))[T:], color='orange', alpha=0.3)
ax2.plot(rbpf.times[T:], gm[T:])
ax2.fill_between(rbpf.times[T:], (gm-1.96*np.sqrt(gv))[T:], (gm+1.96*np.sqrt(gv))[T:], color='orange', alpha=0.3)
ax3.plot(rbpf.times[T:], mm[T:])
ax3.fill_between(rbpf.times[T:], (mm-1.96*np.sqrt(mv))[T:], (mm+1.96*np.sqrt(mv))[T:], color='orange', alpha=0.3)


ax1.set_xticks([])
ax2.set_xticks([])
fig.legend()
plt.show()
