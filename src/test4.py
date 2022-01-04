import numpy as np

P = 2
B =  np.vstack([np.eye(P),
						np.zeros((1, P))])

print(B)