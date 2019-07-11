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
		"The parameters indicate:"
		#Size         = sizes of the square on which we construct the process.
		#                If multidimensional, size is supposed to be a VECTOR!
		# Ihn_function = spatial rate/intensity/inhomogeneity of the PPP.
		self.size  = size
		self.dim   = np.size(size)
		self.inh_f = inh_function

		# We need an upper bound for the function for our algorithm.
		# If not provided we simply assume it is bounded by 1!!
		if 'max_f' in kwargs:
			self.max_f = kwargs['max_f']
		else:
			self.max_f = 1

	def generate(self):
		"Generate a homogeneous PPP"

		# 1) Total number of particles
		self.total  = np.random.poisson(lam = np.prod(self.size))
		# 2) Conditional on total number, sample independent, uniform points
		self.points = np.random.uniform(low = np.zeros(shape = self.dim), high = self.size, size = [self.total, self.dim])

		# Then we generate the inhomogeneous one, by deleting the ones which have low probability of happening.

		# We can do this, unless we have no points to delete.
		if self.points.size:
			self.likely = self.inh_f(self.points)/self.max_f
			self.accept = np.random.binomial(n=1, p = self.likely, size = self.	total)
			# We transform the result to Boolean values
			self.accept = [ _==1 for _ in self.accept ]

			self.points = self.points[self.accept, :]

		return self.points

def input_f( input ):
	return 0.5

# This is to try out the code:
# Below the signature indicates that this should map vectors to numbers

#vinput_f = np.vectorize(input_f, signature = '(m)->()')
#prova = inh_ppp(size = [1.0, 1.0, 1.0, 2.0], inh_function = vinput_f)