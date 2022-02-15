import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pandas as pd
from datahandler import TimeseriesData
from process import *
from particlefilter import RBPF


### --- Forward Simulation --- ###


lss = LangevinModel(mu=0., sigmasq=1., beta=0.8, kv=0.05, theta=-0.8, gsamps=10_000)
lss.generate(nobservations=200)


## - store data in a dataframe - ##
sampled_dic = {'Date_Time': lss.observationtimes, 'Price': lss.observationvals}
sampled_data = pd.DataFrame(data=sampled_dic)

## - option to plot simulated data - ##

# fig = plt.figure()
# ax = fig.add_subplot()
# ax.plot(lss.observationtimes, lss.observationvals)
# ax.set_xticks([])
# plt.show()



### --- importing data --- ###

## - import data from a .csv - ##

# data = TimeseriesData(os.pardir+"/resources/data/test_data.csv")
# df_u = data.remove_non_unique(ret=True)
# plt.plot(df_u['Time'], df_u['Price'])
# plt.xticks([])
# plt.show()




### --- RBPF --- ###


## - define particle filter - ##

# rbpf = RBPF(mumu=0., sigmasq=1., beta=0.8, kw=1e6, kv=.05, theta=-0.8, data=sampled_data, N=1_000, gsamps=10_000, epsilon=0.5)
# ## - containers for storing results of rbpf - ##
# fig = plt.figure()
# ax1 = fig.add_subplot(211)
# ax2 = fig.add_subplot(212)

# ## - main loop of rbpf - ##
# sm, sv, gm, gv, lml = rbpf.run_filter(ret_history=True)

# ## - plotting results of rbpf - ##
# ax1.plot(rbpf.times, lss.observationvals-lss.observationvals[0], label='true')
# ax1.plot(rbpf.times, sm)
# ax1.fill_between(rbpf.times, sm-1.96*np.sqrt(sv), sm+1.96*np.sqrt(sv), color='orange', alpha=0.3)

# ax2.plot(rbpf.times, lss.observationgrad, label='true')
# ax2.plot(rbpf.times, gm)
# ax2.fill_between(rbpf.times, gm-1.96*np.sqrt(gv), gm+1.96*np.sqrt(gv), color='orange', alpha=0.3)


# ax1.set_xticks([])
# ax2.set_xticks([])
# fig.legend()
# plt.show()



### --- Parameter Estimation --- ###
thetas = np.linspace(-1.5, -0.1, 11)
print(thetas)
lmls = []
for theta in tqdm(thetas):
	rbpf = RBPF(mumu=0., sigmasq=1., beta=0.8, kw=1e6, kv=.05, theta=theta, data=sampled_data, N=1_000, gsamps=5_000, epsilon=0.5)
	lml = rbpf.run_filter()
	lmls.append(-1.*lml)

fig = plt.figure()
ax = fig.add_subplot()
ax.bar(thetas, lmls, width=0.05)
ax.set_xlabel('theta')
ax.set_ylabel('-lml')
plt.show()
