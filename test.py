import numpy as np
import pandas as pd
from gamma_proc import gen_gamma_process, marginal_gamma
from ts_proc import gen_ts_process
from functions import *
import matplotlib.pyplot as plt

ALPHA = 1.0
BETA = 1.0
C = 10.0
T = 1.0

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

# generate_and_plot(lambda: gen_gamma_process(C, BETA, 1.0, 1000, maxT=T), 10000, gamma_marginal=lambda: marginal_gamma(np.linspace(0, 30, 1000), T, C, BETA))
# generate_and_plot(lambda: gen_ts_process(ALPHA, C, BETA, 1.0, 1000, maxT=T), 100)

df = pd.read_csv('./test_data.csv')
times = np.flip(df["Time"].values)
prices = np.flip(df["Price"].values)

times_u, indices_u = np.unique(times, return_index=True)
prices_u = np.take_along_axis(prices, indices_u, axis=0)


prices_u_s = moving_average(prices_u, 3)
prices_u_s = np.pad(prices_u_s, (1, 1), 'edge')

plt.subplot(211)
plt.title('Smoothed Price Data - Non-unique ticks removed. Kernel Size = 3')
plt.plot(times_u, prices_u)
plt.xticks([])
plt.xlabel('time in seconds - 2h of data')
plt.subplot(212)
plt.plot(times_u, prices_u_s)
plt.xticks([])
plt.xlabel('time in seconds - 2h of data')

plt.show()