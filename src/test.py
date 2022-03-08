import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pandas as pd
from datahandler import TimeseriesData
from process import *
from particlefilter import RBPF
from matplotlib.colors import to_rgb
from scipy.stats import invgamma
from scipy.special import gamma

plt.style.use('ggplot')

### --- Forward Simulation --- ###


lss = LangevinModel(x0=1., xd0=0., mu=2., sigmasq=1., beta=1.8, kv=1e-6, kmu=1e-2, theta=-2., gsamps=5_000)
lss.generate(nobservations=500)


## - store data in a dataframe - ##
sampled_dic = {'Date_Time': lss.observationtimes, 'Price': lss.observationvals}
sampled_data = pd.DataFrame(data=sampled_dic)

## - option to plot simulated data - ##

fig = plt.figure()
ax1 = fig.add_subplot(311)
ax1.plot(lss.observationtimes, lss.observationvals)
# ax1.set_xticks([])

ax2 = fig.add_subplot(312)
ax2.plot(lss.observationtimes, lss.observationgrad)
# ax2.set_xticks([])

ax3 = fig.add_subplot(313)
ax3.plot(lss.observationtimes, lss.observationmus)

plt.show()


### --- RBPF --- ###


## - define particle filter - ##

rbpf = RBPF(mux=1., mumu=0., beta=1.8, kw=2., kv=1e-6, kmu=1e-2, rho=1e-5, eta=1e-5, theta=-2., data=sampled_data, N=250, gsamps=100, epsilon=0.5)
## - containers for storing results of rbpf - ##
fig = plt.figure()
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)
## - main loop of rbpf - ##
sm, sv, gm, gv, mm, mv, lml, count, weights, Es = rbpf.run_filter(ret_history=True)

T = 0

## - plotting results of rbpf - ##
ax1.plot(rbpf.times[T:], lss.observationvals[T:], label='true')
ax1.plot(rbpf.times[T:], sm[T:])
ax1.fill_between(rbpf.times[T:], (sm-1.96*np.sqrt(1.*sv))[T:], (sm+1.96*np.sqrt(1.*sv))[T:], color='orange', alpha=0.3)

ax2.plot(rbpf.times[T:], lss.observationgrad[T:], label='true')
ax2.plot(rbpf.times[T:], gm[T:])
ax2.fill_between(rbpf.times[T:], (gm-1.96*np.sqrt(gv))[T:], (gm+1.96*np.sqrt(gv))[T:], color='orange', alpha=0.3)

ax3.plot(rbpf.times[T:], lss.observationmus[T:], label='true')
ax3.plot(rbpf.times[T:], mm[T:])
ax3.fill_between(rbpf.times[T:], (mm-1.96*np.sqrt(mv))[T:], (mm+1.96*np.sqrt(mv))[T:], color='orange', alpha=0.3)

# ## - prediction - ##
# final_time = rbpf.prev_time
# at = 0.
# ct = 0.
# at2 = 0.
# ct2 = 0.
# for particle in rbpf.particles:
# 	w = np.exp(particle.logweight)
# 	ap, Cp = particle.predict(final_time, final_time+1.)
# 	at += (ap*w)[0,0]
# 	ct += (Cp*w)[0,0]
# 	at2 +=(ap*w)[1,0]
# 	ct2 += (Cp*w)[1,1]
# ts = np.array([final_time, final_time+1.])
# pm = np.array([sm[-1], at])
# pv = np.array([sv[-1], ct])
# pm2 = np.array([gm[-1], at2])
# pv2 = np.array([gv[-1], ct2])
# # print(pm, pv)
# ax1.plot(ts, pm, ls='--')
# ax1.fill_between(ts, pm-1.96*np.sqrt(1.*pv), pm+1.96*np.sqrt(1.*pv), color='red', alpha=0.3, ls='--')





# ax2.plot(rbpf.times[-100:], lss.observationgrad[-100:], label='true')
# ax2.plot(rbpf.times[-100:], gm[-100:])
# ax2.fill_between(rbpf.times[-100:], (gm-1.96*np.sqrt(gv))[-100:], (gm+1.96*np.sqrt(gv))[-100:], color='orange', alpha=0.3)
# ax2.plot(ts, pm2, ls='--')
# ax2.fill_between(ts, pm2-1.96*np.sqrt(1.*pv2), pm2+1.96*np.sqrt(1.*pv2), color='red', alpha=0.3, ls='--')

## - full history - ##
# states, grads = rbpf.run_filter_full_hist()
# ax1.plot(rbpf.times, lss.observationvals-lss.observationvals[0], lw=1.5)
# ax1.plot(rbpf.times, states, alpha=0.05, ls='-')
# ax2.plot(rbpf.times, lss.observationgrad-lss.observationgrad[0], lw=1.5)
# ax2.plot(rbpf.times, grads, alpha=0.05, ls='-')

# ax1.set_xticks([])
# ax2.set_xticks([])
fig.legend()

fig2 = plt.figure()
axxx = fig2.add_subplot(111)
axis = np.linspace(0.1, 2., 1000)

def pdf(x, rho, eta):
	return (np.power(eta, rho)/gamma(rho)) * np.power(x, (-rho-1))*np.exp(-eta/x)
mixture = np.zeros(1000)
mixture = pdf(axis, 0.1, 0.1)
# print(weights)
for i in range(200):
	print(count/2, Es[i])
	mixture += (weights[i] * pdf(axis, 1.+(count/2.), 1.+(Es[i]/2.))).flatten()
axxx.plot(axis, mixture)
plt.show()



### --- Parameter Estimation --- ###
# thetas = np.linspace(-5., -0.2, 10)
# print(thetas)
# lmls = []
# for theta in tqdm(thetas):
# 	rbpf = RBPF(mumu=0., sigmasq=1., beta=0.8, kw=1e6, kv=.05, theta=theta, data=sampled_data, N=1_000, gsamps=5_000, epsilon=0.5)
# 	lml = rbpf.run_filter()
# 	lmls.append(-1.*lml)

# fig = plt.figure()
# ax = fig.add_subplot()
# ax.bar(thetas, lmls, width=0.05)
# ax.set_xlabel('theta')
# ax.set_ylabel('-lml')
# plt.show()
