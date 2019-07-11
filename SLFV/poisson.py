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

# We define a class which allows us to simulate and arbitrary inhomogeneous
# Poisson point process.


class inh_ppp:
	def __init__(self, size, inh_function, **kwargs):
		# The parameters indicate:
		# Size         = sizes of the square on which we construct the process.
		#                If multidimensional, size is supposed to be a VECTOR!
		# Ihn_function = spatial rate/intensity/inhomogeneity of the PPP.
		self.size  = size
		self.dim   = np.shape(size)
		self.inh_f = inh_function

		# We need an upper bound for the function for our algorithm.
		# If not provided we simply assume it is bounded by 1!!
		if 'max_f' in kwargs:
			self.max_f = kwargs['max_f']
		else:
			self.max_f = 1

	def generate(self):
		# First we generate a homogeneous PPP.

		# 1) Total number of particles
		self.total  = np.random.poisson(lam = np.prod(self.size), size = 1)
		# 2) Conditional on total number, sample independent, uniform points
		self.points = np.random.uniform(low = np.zeros(size = self.dim), high = self.size, size = [self.total, self.dim])

		# Then we generate the inhomogeneous one, by deleting the ones which have low probability of happening.

		self.likely   = self.inh_f(self.points)/self.max_f
		self.tocancel = 1 - np.random.binomial(n=1, p = self.likely, shape = self.total)

		self.points = self.points[self.tocancel, :]

		return self.points


time_horizon = 10
box_size     = 10

# Simulate a Poisson Point Process.
total_num_events = np.random.poisson(lam = box_size**2)