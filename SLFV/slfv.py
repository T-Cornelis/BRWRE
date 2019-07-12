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

# To time stuff we import the measure decorator from my_timeit.py
from my_timeit import measure

class slfv:
	def __init__(self, DIM, SIZE, TIME_HORZ, neu_f, sel_p, sel_n, initial_f, **kwargs):
		"""
		DIM = dimension of the space.
		SIZE = size of the box.
		TIME_HORZ = time horizon.
		neu_f = neutral inhomogeneity function, values in (0,1).
		sel_p = positive selection, values in (0,1).
		sel_n = negative selection, values in (0,1).
		initial_f = initial state of the process (takes values in (0,1)).
		_________
		Kwargs:
		MESH_PRECISION = distance between two points on the discrete mesh
		"""

		self.DIM   = DIM
		self.SIZE  = SIZE
		self.TIME_HORZ = TIME_HORZ
		self.neu_f = neu_f
		self.sel_p = sel_p
		self.sel_n = sel_n
		self.initial_f  = initial_f

		if 'MESH_PRECISION' in kwargs:
			self.MESH_PRECISION = kwargs['MESH_PRECISION']
		else:
			self.MESH_PRECISION = 1e-2

		# We also define a vector containing all sizes
		self.vsize = self.SIZE*np.ones(shape = DIM+1)
		self.vsize[-1] = self.TIME_HORZ

		# We define and initialize the neutral events:
		self.pois_neu    = ps.inh_ppp(size = self.vsize, inh_function = self.neu_f)
		self.e_neu       = self.pois_neu.generate()

		# Similarly for positive selection events:
		self.pois_sel_p  = ps.inh_ppp(size = self.vsize, inh_function = self.sel_p)
		self.e_sel_p     = self.pois_sel_p.generate()
		
		# ... and negative selection
		self.pois_sel_n  = ps.inh_ppp(size = self.vsize, inh_function = self.sel_n)
		self.e_sel_n     = self.pois_sel_n.generate()

		# We then create a discretization of the model:
		self.discretize()


	def discretize(self):
		"""
		We create a discretization of the initial condition.
		WE ASSUME THAT THE SPATIAL SIZE IS THE SAME IN EVERY DIRECTION
		(i.e. the domain is a square)
		"""
		grid = np.arange(start = 0.0, stop = self.vsize[0], step = self.MESH_PRECISION)
		grid_list = [grid]*self.DIM

		# We create the meshgrid and initialize the function on this grid.
		# Note that mesh.T (in array form) fits well with row vectorization of
		# the form '(n)->()'
		self.mesh = np.array(np.meshgrid(*grid_list)).T
		# In this mesh:
		# x axis <-> columns
		# y axis <-> rows
		self.cur_value = self.initial_f(self.mesh)		

	def generate(self):
		"""
		Generates a new set of neutral/selective points and orders them in time.
		"""
		self.e_neu   = self.pois_neu.generate()
		self.e_sel_p = self.pois_sel_p.generate()
		self.e_sel_n = self.pois_sel_n.generate()

		# Then we order the events:
		#self.order   = 

	def do_step(self):
		"""
		Goes one step forward in the process.
		"""


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

#3) We define the initial condition:
def initial_f(input):
	return np.abs( 0.2* np.cos(np.dot(input, input)))
vinitial_f = np.vectorize(initial_f, signature = '(n)->()')

prova = slfv(DIM, SIZE, TIME_HORZ, vneutral, vselection_pl, vselection_mn, vinitial_f)



