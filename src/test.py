import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pandas as pd
from datahandler import TimeseriesData
from process import *
from particlefilter import RBPF


### --- Forward Simulation --- ###


# lss = LangevinModel(mu=0., sigmasq=1., beta=0.1, kv=0.05, theta=-0.8, gsamps=10_000)
# lss.generate(nobservations=500)


## - option to plot simulated data - ##

# fig = plt.figure()
# ax = fig.add_subplot()
# ax.plot(lss.observationtimes, lss.observationvals)
# ax.set_xticks([])
# plt.show()


## - store data in a dataframe - ##

# sampled_dic = {'Date_Time': lss.observationtimes, 'Price': lss.observationvals}
# sampled_data = pd.DataFrame(data=sampled_dic)




### --- importing data --- ###

## - import data from a .csv - ##

# data = TimeseriesData(os.pardir+"/resources/data/test_data.csv")
# df_u = data.remove_non_unique(ret=True)
# plt.plot(df_u['Time'], df_u['Price'])
# plt.xticks([])
# plt.show()




### --- RBPF --- ###


## - define particle filter - ##

rbpf = RBPF(mumu=0., sigmasq=1., beta=0.1, kw=1e6, kv=.1, theta=-0.5, data=sampled_data, N=N, gsamps=5000, epsilon=0.8)

## - containers for storing results of rbpf - ##
fig = plt.figure()
ax = fig.add_subplot()
state_means = []
state_variances = []


## - main loop of rbpf - ##
for _ in tqdm(range(len(sampled_data)-1)):
	rbpf.increment_particles()
	rbpf.reweight_particles()
	state_means.append(rbpf.get_state_mean()[0][0])
	state_variances.append(rbpf.get_state_covariance()[0,0])
	if rbpf.get_logDninf() < rbpf.log_resample_limit:
		rbpf.resample_particles()


## - plotting results of rbpf - ##
ax.plot(lss.observationtimes, lss.observationvals, label='true')
ax.plot(lss.observationtimes[:-1], state_means+rbpf.initial_price)
ax.fill_between(lss.observationtimes[:-1], state_means+rbpf.initial_price-1.96*np.sqrt(state_variances), state_means+rbpf.initial_price+1.96*np.sqrt(state_variances), color='orange', alpha=0.3)

ax.set_xticks([])
fig.legend()
plt.show()
