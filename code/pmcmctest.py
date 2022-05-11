import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from datahandler import TimeseriesData
from process import *
from particlefilter import RBPF
from pmcmc import PMMH


plt.style.use('ggplot')

### --- Forward Simulation --- ###

x0 = 0.
xd0 = 0.
mu0 = 0.
sigmasq = 1.
beta = 5.
kv = 5e-4
kmu = 1e-5
theta = -2.
p = 0.
gsamps1 = 5_000

kw = 2.
rho = 1e-5
eta = 1e-5
N = 200
gsamps2 = 200
epsilon = 0.5

lss = LangevinModel(x0=x0, xd0=xd0, mu=mu0, sigmasq=sigmasq, beta=beta, kv=kv, kmu=kmu, theta=theta, p=p, gsamps=gsamps1)
lss.generate(nobservations=100)


## - store data in a dataframe - ##
sampled_dic = {'Telapsed': lss.observationtimes, 'Price': lss.observationvals}
sampled_data = pd.DataFrame(data=sampled_dic)

## - option to plot simulated data - ##

# fig = plt.figure()
# ax1 = fig.add_subplot(311)
# ax1.plot(lss.observationtimes, lss.observationvals)
# # ax1.set_xticks([])

# ax2 = fig.add_subplot(312)
# ax2.plot(lss.observationtimes, lss.observationgrad)
# # ax2.set_xticks([])

# ax3 = fig.add_subplot(313)
# ax3.plot(lss.observationtimes, lss.observationmus)

# ax1.set_ylabel(r'Position, $x$')
# ax2.set_ylabel(r'Velocity, $\.x$')
# ax3.set_ylabel(r'Skew, $\mu$')
# plt.show()

### --- Metropolis-Hastings --- ###

pmmh = PMMH(mux=0., mumu=0., kw=kw, rho=rho, eta=eta, data=sampled_data, N=N, gsamps=gsamps2, epsilon=epsilon, delta=0.1, sampleX=True)
x, phis = pmmh.run_sampler(500)
x = np.array(x)[100:]
phis = np.array(phis)[100:]
print(np.mean(phis, axis=0))