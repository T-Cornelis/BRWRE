# We model BRWRE in 1D

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

# We import all the function from the header:

cdef extern from "my_list.h":

	ctypedef struct LS_element:
		double* LS_value_dbl;
		int* LS_value_int;
		LS_element * LS_next;

	ctypedef struct List:
		LS_element * First
		LS_element * Last
		size_t dim

cdef extern from "my_list.c":

	void INI_list(List * data, double * vector_dbl, int* vector_int, int dim)

cdef extern from "brwre_black.c":

	ctypedef struct Brwre:
		int count
		double* spatial_noise
		int total
		List* data
		double* times
		int ERROR_NUMBER
		double aim 
		int move_on
		int END_reached
		LS_element* to_move
		LS_element* previous
		double cur_particle

	void INI_brwre(Brwre* particle, double* spat_noise, double* times, List* ini_pos, double aim)

	void REPEAT_move(Brwre* particle)

# We Define the class for the solution to the SHE

class she:
	"""
	We model PAM.
	"""
	def __init__(self, x0, noise, delta_t):
		# Initial state of the sistem.
		self.state = x0
		self.noise = noise
		self.delta_t = delta_t

	def do_step(self):
		# We do one more step in the implicit Euler approximations
		self.state = np.dot(resolvent, self.state) + self.delta_t*np.dot(resolvent, np.multiply(self.state, self.noise))
		#self.state = np.dot(resolvent, self.state  + np.multiply(self.state, self.noise) )
		#self.state = np.dot(resolvent, self.state + np.random.normal(size = (space_pts), scale = np.sqrt(delta_t/delta_x) ) ) 


# HAVE TO CHANGE THE TWO VALUES BELOW ALSO IN C FILES!
# Number of particles at the beginning.
cdef int INI_PART

INI_PART = 10000

# Size of the box:
cdef int CENTER
cdef int DIM_BOX

DIM_BOX = 10000
CENTER  = 5000

# Number of random times we generate at once:
cdef int DIM_RANDOM

DIM_RANDOM = 100000

# SPATIAL_NOISE is the (random) potential the particles move in.
cdef double SPATIAL_NOISE[10000]
# We introduce an additional intensity constant, otherwise the effects are too small.
intensity = 15
# This is the true macroscopic noise.
noise_macro = np.random.normal(size = (DIM_BOX))*np.sqrt(DIM_BOX)*intensity
# This noise is scaled w.r.t to the time of the particle system (so it's much weaker)
SPATIAL_NOISE = noise_macro/DIM_BOX**2

# TIME_NOISE is the number of random times we generate.
cdef double TIME_NOISE[100000]
TIME_NOISE = np.random.exponential(size = (DIM_RANDOM))

# We create our BRWRE sample:
cdef Brwre sample
cdef Brwre* particle

particle = &sample

# We create the intializing list:

cdef List initial_list
cdef List* ini_ptr

ini_ptr = &initial_list

cdef double initial_dbl[10000]
cdef int initial_int[10000][2]

initial_times  = np.random.exponential(size = (INI_PART))
initial_choice = np.random.binomial(1, 0.5, size= (INI_PART))

for i in range(INI_PART):
	initial_dbl[i] = initial_times[i]
	initial_int[i][0] = i
	initial_int[i][1] = initial_choice[i]

INI_list(ini_ptr, &initial_dbl[0], &initial_int[0][0], INI_PART)

# Describes how much time we let pass for every frame.
time_step = 0.00002
aim = time_step*(DIM_BOX**2)

# So that now we can initialize the BRWRE sample:
INI_brwre(particle, &SPATIAL_NOISE[0], &TIME_NOISE[0], ini_ptr, aim)


# Parameters for the sizes of the picture

B_NUM = 3000
B_NUM_EXTRA = 50

# Compute the spatial subdivision.
# And start the brwre_histogram.

cdef LS_element* temp_list

cur_dim = particle[0].data[0].dim
temp_list = particle[0].data[0].First
python_loc = np.zeros(shape = (cur_dim))
for i in range(cur_dim):
	python_loc[i] = temp_list[0].LS_value_int[0]
	temp_list = temp_list[0].LS_next

brwre_histo, space_histo = np.histogram(python_loc, bins = B_NUM, range = (0, DIM_BOX), density = False)
space = space_histo[:-1]

space_pts = len(space)

# Compute the noise average (a "ball average integral" of the noise):
noise_avrg = np.zeros(shape = space_pts)
for i in range(space_pts):
	for m in xrange( int(space_histo[i]), int(space_histo[i+1]) ):
		noise_avrg[i] += noise_macro[m]

#increment  = np.floor(DIM_BOX/B_NUM)
#for i in range(space_pts):
#	for _ in range( int(increment) ):
#		noise_avrg[i] += noise_macro[i+_]

# Below, multiply with B_NUM is the average in space.
# Every box is macroscopically of size 1/B_NUM
# Divide with DIM_BOX is because of the Riemann integral:
# \int \xi^n(x)f(x) = \sum_x n^{-1} \xi^n(x) f(x)

noise_avrg = noise_avrg*(B_NUM/DIM_BOX)

# Time-space discretisation for PAM
delta_t = time_step
delta_x = 1/space_pts

# This is the resolvent of the periodic laplacian (real laplacian, no coefficient in front) matrix:
# to_invert = (Id - delta_t* Delta)^-1

main_diag = np.ones(shape = (space_pts))
offu_diag = 0.5*(1/(1+2*(delta_t/delta_x**2)) -1)*np.ones(shape = (space_pts-1))
to_invert = scipy.sparse.diags([offu_diag, main_diag, offu_diag], [-1, 0, 1]).toarray()
to_invert[0,space_pts-1] = 0.5*(1/(1+2*(delta_t/delta_x**2)) -1)
to_invert[space_pts-1, 0] = 0.5*(1/(1+2*(delta_t/delta_x**2)) -1)
resolvent = scipy.linalg.inv(to_invert)/(1+2*(delta_t/delta_x**2))

# We create a sample path for PAM
# with initial condition the same as the histogram:
she_sample = she(brwre_histo*B_NUM/DIM_BOX, noise_avrg, delta_t)

# We redefine the histogram (a smaller one):
brwre_histo, space_histo = np.histogram(python_loc, bins = B_NUM_EXTRA, range = (0, DIM_BOX), density = False)
space_extra = space_histo[:-1]
# We define the animation function:
def animate(i):

	global cur_dim, aim, TIME_NOISE, brwre_histo

	# Real time is:
	ani_time = i*delta_t
	
	# Redefine the plot for PAM
	lines_1.set_data(space/DIM_BOX, she_sample.state)
	# Redefine the plot for the particles
	lines_2.set_data(space_extra/DIM_BOX, brwre_histo*B_NUM_EXTRA/DIM_BOX)

	# Set the new time
	time_text.set_text("Time = {:2.3f}, Num prtcls: {}, Rho = {}".format(ani_time, cur_dim, 1.0) )
	# We print the step we are in:
	sys.stdout.flush()
	sys.stdout.write("\r Step = {}".format(i))
	
	# And we do the next step for PAM:
	she_sample.do_step()

	# We reset the times for the particle:
	error = 2
	count_steps = 0
	TIME_NOISE = np.random.exponential(size = (DIM_RANDOM))
	particle[0].times = &TIME_NOISE[0]
	particle[0].count = 0
	particle[0].ERROR_NUMBER = 0
	particle[0].END_reached = 0
	particle[0].move_on = 0
	particle[0].to_move = particle[0].data.First
	particle[0].previous = NULL
	particle[0].cur_particle = 1

	# We set the new aim for the particles
	particle[0].aim = (i+1)*aim

	# We do the next step in the particle system:
	start_time_out = time.time()
	while (error == 2):
		
		start_time = time.time()

		REPEAT_move(particle)
			
		if(particle[0].ERROR_NUMBER == 2):
			TIME_NOISE = np.random.exponential(size = (DIM_RANDOM))
			particle[0].times = &TIME_NOISE[0]
			particle[0].count = 0
			particle[0].ERROR_NUMBER = 0

		else:
			error = 0

		sys.stdout.flush()
		sys.stdout.write("\r Step = {}. This step took me {:2.3f} s. Percentual on total: {:1.5f}. Elapsed total: {:4.2f}. Hrs to end:{:4.2f}".format( \
			count_steps, time.time()-start_time, particle[0].cur_particle/particle[0].data[0].dim, \
			time.time()-start_time_out, \
			(time.time()-start_time_out)*(particle[0].data[0].dim-particle[0].cur_particle)/(particle[0].cur_particle*3600) ) )
		
		count_steps += 1	

	print("\n \n ")
	sys.stdout.flush()
	sys.stdout.write("Finished frame {} in {:2.3f} seconds".format(i, time.time()-start_time_out))
	print("\n \n")

	# Pass the values to Python

	cur_dim = particle[0].data[0].dim
	temp_list = particle[0].data[0].First
	python_loc = np.zeros(shape = (cur_dim))
	for i in range(cur_dim):
		python_loc[i] = temp_list[0].LS_value_int[0]
		temp_list = temp_list[0].LS_next

	# New histogram:
	brwre_histo, space_histo = np.histogram(python_loc, bins = B_NUM_EXTRA, range = (0, DIM_BOX), density = False)
	
	return [lines_1,] + [lines_2,] + [time_text,]


#We set up the picture
fig       = plt.figure()
ax        = plt.axes(xlim=(0, 1), ylim = (0.5, 1.5))
time_text = ax.text(0.05, 0.95,'',horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
lines_1,  = ax.plot([],[], lw = 2, color = "green")
lines_2,  = ax.plot([],[], lw = 2, color = "red")

plt.title("Branching RW in random environment") 

embed()

# We let the animation go.
ani       = animation.FuncAnimation(fig, animate, frames=30, interval = 70, blit = True)
ani.save(filename = 'brwre_pam.mp4', extra_args=['-vcodec', 'libx264'], bitrate = 7000)

#END