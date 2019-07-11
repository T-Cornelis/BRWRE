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

# For now we do not ad any selection.

time_horizon = 10
box_size     = 10

# Simulate a Poisson Point Process.
total_num_events = np.random.poisson(lam = box_size**2)