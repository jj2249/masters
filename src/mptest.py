import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

import os
# from pathos.multiprocessing import ProcessingPool
from p_tqdm import p_map
# import multiprocessing as mp

from process import *
from datahandler import TimeseriesData
from particlefilter import RBPF


def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())



if __name__ == '__main__':
	info('main line')

	lss = LangevinModel(mu=0., sigmasq=1., beta=0.8, kv=0.05, theta=-1.5, gsamps=10_000)
	lss.generate(nobservations=200)


	## - store data in a dataframe - ##
	sampled_dic = {'Date_Time': lss.observationtimes, 'Price': lss.observationvals}
	sampled_data = pd.DataFrame(data=sampled_dic)

	## - option to plot simulated data - ##

	fig = plt.figure()
	ax = fig.add_subplot()
	ax.plot(lss.observationtimes, lss.observationvals)
	ax.set_xticks([])
	plt.show()

	thetas = np.linspace(-3., -0.1, 16)
	# thetas = np.array([-1.5])
	
	rbpf = RBPF(mumu=0., beta=0.8, kw=1e-5, kv=.05, theta=-1.5, data=sampled_data, N=500, gsamps=5_000, epsilon=0.5)
	lmls = p_map(rbpf.run_filter_MP, thetas)

	fig = plt.figure()
	ax = fig.add_subplot()
	ax.bar(thetas, lmls, width=(3.0-0.1)/(5*32.))
	ax.set_xlabel('theta')
	ax.set_ylabel('lml')
	plt.show()