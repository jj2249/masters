import numpy as np
import matplotlib.pyplot as plt
import os

import statsmodels.api as sm

import src.datahandler as dth
plt.style.use('seaborn')

# tobj = dth.TimeseriesData(os.pardir+"/resources/data/oildata2.csv", idx1=0)
tobj = dth.TickData(os.pardir+"/resources/data/EURGBP-2022-04.csv", nrows=500)
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
tobj.plot(ax1)
print(tobj.df)
diffs = tobj.df[['Bid','Ask']].diff().iloc[1:]
print(diffs)
# incs = diffs['Bid']/tobj.df['DeltaTs'].iloc[1:]
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.hist(tobj.df['Bid'], bins=50, density=True)

# fig3 = plt.figure()
# ax3 = fig3.add_subplot(111)
# sm.qqplot(incs, fit=True, line='s', ax=ax3)
# ax3.set_ylim(-4., 4.)
plt.show()