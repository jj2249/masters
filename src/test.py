import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pandas as pd
from datahandler import TimeseriesData
from process import *
from particlefilter import RBPF


### --- Forward Simulation --- ###
lss = LangevinModel(muw=0., kv=0., theta=-0.1, C=1., beta=1., nobservations=50)
lss.forward_simulate()
# plt.plot(lss.observationtimes, lss.observationvals)
# plt.xticks([])
# plt.show()

sampled_dic = {'Date_Time': lss.observationtimes, 'Price': lss.observationvals}
sampled_data = pd.DataFrame(data=sampled_dic)
rbpf = RBPF(P=2, mumuw=0., kw=1e12, kv=0., theta=-0.1, C=1., beta=1., data=sampled_data, N=1)

### --- importing data --- ### 
# data = TimeseriesData(os.pardir+"/resources/data/test_data.csv")
# df_u = data.remove_non_unique(ret=True)
# print(df_u)
# plt.plot(df_u['Time'], df_u['Price'])
# plt.xticks([])
# plt.show()


### --- RBPF --- ###
# data = TimeseriesData(os.pardir+"/resources/data/test_data.csv")
# df_u = data.remove_non_unique(ret=True)
# rbpf = RBPF(P=2, mumuw=1., kw=1e6, kv=1., theta=-1, C=10., beta=.1, data=df_u, N=1000)
state_means = []
state_variances = []
for i in tqdm(range(len(sampled_data)-1)):
	rbpf.increment_particles()
	rbpf.reweight_particles()
	state_means.append(rbpf.get_state_mean()[0][0])
	state_variances.append(rbpf.get_state_covariance()[0,0])
	rbpf.resample_particles()


state_means = np.array(state_means)
state_variances = np.array(state_variances)
fig = plt.figure()
ax = fig.add_subplot()
ax.plot(lss.observationtimes, lss.observationvals, label='true')
ax.errorbar(lss.observationtimes[:-1], state_means+rbpf.initial_price, yerr=1.96*np.sqrt(state_variances))

ax.set_xticks([])
fig.legend()
plt.show()

# test_data = 