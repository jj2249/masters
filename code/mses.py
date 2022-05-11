import numpy as np
import os
from tqdm import tqdm
import pandas as pd
from datahandler import TimeseriesData
from process import *
from particlefilter import RBPF
import matplotlib as mpl
# mpl.use("pgf")
# mpl.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False
# })

from p_tqdm import p_umap
from functools import partial

import matplotlib.pyplot as plt

plt.style.use('seaborn')

def get_mse(x0, mu0, beta, kw, kv, kmu, rho, eta, theta, p, sampled_data, gsamps, epsilon, N):
    rbpf = RBPF(mux=x0, mumu=mu0, beta=beta, kw=kw, kv=kv, kmu=kmu, rho=rho, eta=eta, theta=theta, p=p, data=sampled_data, N=N, gsamps=gsamps, epsilon=epsilon)
    _, _, _, _, _, _, _, _, _, MSEs = rbpf.run_filter(ret_history=True, plot_marginal=False, tpred=0., progbar=False)
    return N, np.sum(MSEs)


if __name__ == "__main__":
    tw = 6.50127
    ### --- Forward Simulation --- ###

    x0 = 0.
    xd0 = 0.
    mu0 = 0.0
    sigmasq = 1.
    beta = 1.
    kv = 1e-10
    kmu = 1e-15
    theta = -2.
    p = 0.
    gsamps1 = 100_000

    kw = 2.
    rho = 1e-5
    eta = 1e-5
    gsamps2 = 500
    epsilon = 0.5
    lss = LangevinModel(x0=x0, xd0=xd0, mu=mu0, sigmasq=sigmasq, beta=beta, kv=kv, kmu=kmu, theta=theta, p=p, gsamps=gsamps1)
    lss.generate(nobservations=100)


    ## - store data in a dataframe - ##
    sampled_dic = {'Telapsed': lss.observationtimes, 'Price': lss.observationvals}
    sampled_data = pd.DataFrame(data=sampled_dic)

    ## - option to plot simulated data - ##

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.plot(lss.observationtimes, lss.observationvals)
    # ax1.set_xticks([])
    # ax1.xaxis.set_tick_params(length=0)
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax2 = fig.add_subplot(212)
    ax2.plot(lss.observationtimes, lss.observationgrad)
    # ax2.set_xticks([])

    ax1.set_ylabel(r'Position, $x$')
    ax2.set_ylabel(r'Velocity, $\dot{x}$')
    ax2.set_xlabel(r'Time, $t$')
    # plt.show()
    fig.set_size_inches(w=tw, h=0.5*tw)
    plt.tight_layout()
    # plt.savefig('../resources/figures/bmsde.pgf')
    plt.show()
    ### --- RBPF --- ###


    ### -- Mean square errors -- ###
    Ns = np.arange(50, 1000, 50)
    results = np.array(p_umap(partial(get_mse, x0, mu0, beta, kw, kv, kmu, rho, eta, theta, p, sampled_data, gsamps2, epsilon), Ns))
    print(results)
    idx = np.argsort(results[:,0])
    ns = np.take(results[:,0], idx)
    results = np.take(results[:,1], idx)

    plt.plot(ns, results)
    plt.show()