import numpy as np
from gamma_proc import gen_gamma_process
from ts_proc import gen_ts_process
from functions import *

ALPHA = 1.0
BETA = 1.0
C = 10.0
T = 1.0


generate_and_plot(lambda: gen_gamma_process(C, BETA, 1.0, 1000, maxT=1.0), 100)
generate_and_plot(lambda: gen_ts_process(ALPHA, C, BETA, 1.0, 1000, maxT=1.0), 100)
