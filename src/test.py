import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pandas as pd
from datahandler import TimeseriesData
from process import *
from particlefilter import RBPF

N = 1_000
T = 100

### --- Forward Simulation --- ###

# fig = plt.figure()
# ax = fig.add_subplot()
lss = LangevinModel(mu=0., sigmasq=1., beta=0.1, kv=0.1, theta=-0.5, nobservations=100)
lss.forward_simulate()
# ax.plot(lss.observationtimes, lss.observationvals)
# ax.set_xticks([])
# plt.show()

sampled_dic = {'Date_Time': lss.observationtimes, 'Price': lss.observationvals}
sampled_data = pd.DataFrame(data=sampled_dic)
rbpf = RBPF(P=2, mumu=0., sigmasq=1., beta=0.1, kw=1e6, kv=.1, theta=-0.5, data=sampled_data, N=N)

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
fig = plt.figure()
ax = fig.add_subplot()
state_means = []
state_variances = []
# particle_vals = []
# times = []
for _ in tqdm(range(len(sampled_data)-1)):
	rbpf.increment_particles()
	rbpf.reweight_particles()
	state_means.append(rbpf.get_state_mean()[0][0])
	state_variances.append(rbpf.get_state_covariance()[0,0])

	# for particle in rbpf.particles:
		# particle_vals.append(particle.acc[0][0])
		# times.append(rbpf.current_time)
	rbpf.resample_particles()

# ax.hexbin(times, particle_vals, cmap='viridis', bins=(T-2, 100))
# plt.show()
# print(state_means+rbpf.initial_price)
# print(state_variances)
state_means = np.array(state_means)
state_variances = np.array(state_variances)

ax.plot(lss.observationtimes, lss.observationvals, label='true')

ax.plot(lss.observationtimes[:-1], state_means+rbpf.initial_price)
# print(state_means.shape)
# print(lss.observationtimes.shape)
# ax.errorbar(lss.observationtimes[:-1], state_means+rbpf.initial_price, yerr=1.96*np.sqrt(state_variances))
ax.fill_between(lss.observationtimes[:-1], state_means+rbpf.initial_price-1.96*np.sqrt(state_variances), state_means+rbpf.initial_price+1.96*np.sqrt(state_variances), color='orange', alpha=0.3)
# ax.plot(df_u['Date_Time'][:-1], df_u['Price'][:-1], label='true')
# ax.plot(df_u['Date_Time'][:-1], state_means+rbpf.initial_price, label='prediction')
# ax.fill_between(df_u['Date_Time'][:-1], state_means+rbpf.initial_price+1.96*state_variances, state_means+rbpf.initial_price-1.96*state_variances, alpha=0.3)
# ax.errorbar(df_u['Date_Time'][:-1], state_means+rbpf.initial_price, yerr=1.96*np.sqrt(state_variances))
# ax.set_xticks([])
# fig.legend()
# fig.suptitle('1000 particles')
plt.show()

# test_data = 
