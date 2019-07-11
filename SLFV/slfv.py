 # We model SLFV in 2D

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors
import scipy as sp
import scipy.sparse
import scipy.sparse.linalg
from IPython import embed
from os import path
import sys
import time

# To simulate a (inhomogeneous) Poisson Point Process we import our package:
import poisson as ps

# 0) We define the parameters of the problem:
DIM       = 2
SIZE      = 10
TIME_HORZ = 5

# 1) We define the selection coefficient (which should always take values in (-1, 1) ):
def selection(input):
	return 0.2*np.sin(np.dot(input, input))
def selection_pl(input):
	return np.clip(0.2*np.sin(np.dot(input, input)), a_min = 0, a_max = None)
def selection_mn(input):
	return -np.clip(0.2*np.sin(np.dot(input, input)), a_min = None, a_max = 0)
# which we then vectorize:
vselection    = np.vectorize(selection, signature = '(n)->()')
vselection_pl = np.vectorize(selection_pl, signature = '(n)->()')
vselection_mn = np.vectorize(selection_mn, signature = '(n)->()')

# 2) We define the neutral coefficient:
def neutral(input):
	return 1 - np.abs(selection(input))
# which we also vectorize
vneutral = np.vectorize(neutral, signature = '(n)->()')

# 3) We define the neutral events:
pois_neu = ps.inh_ppp(size = [SIZE, SIZE, TIME_HORZ], inh_function = vneutral)
neutral_events = pois_neu.generate()

# 4) We define the positive selective events:
pois_sel_pl  = ps.inh_ppp(size = [SIZE, SIZE, TIME_HORZ], inh_function = vselection_pl)
selection_pl_events = pois_sel_pl.generate()
pois_sel_mn  = ps.inh_ppp(size = [SIZE, SIZE, TIME_HORZ], inh_function = vselection_mn)
selection_mn_events = pois_sel_mn.generate()
embed()
