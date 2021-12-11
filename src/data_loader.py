import numpy as np
import scipy.io as io
from sys import argv

def mat_to_np(file):
	with open(file+'.mat', 'rb') as f:
		data = io.loadmat(f)
		f.close()
	with open(file+'.npy', 'wb') as f:
		np.save(f, data)
		f.close()


if __name__ == "__main__":
	mat_to_np(argv[1])