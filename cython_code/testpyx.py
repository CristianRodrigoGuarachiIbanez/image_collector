import counter
#from numpy import ndarray, asarray

import pyximport
pyximport.install()
pyximport.install(pyimport = True)

dictionary = {'o, k1': 1, 'o, k2': 0, 'o, k3': 0}

counter.getNonCollisionsOnly(dictionary)