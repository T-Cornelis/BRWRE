 # We model BRWRE in 2D

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

cdef extern from "my_2D_list.h":

	ctypedef struct LS_element:
		double* LS_value_dbl;
		int* LS_value_int;
		LS_element * LS_next;

	ctypedef struct List:
		LS_element * First
		LS_element * Last
		size_t dim

cdef extern from "my_2D_list.c":

	void INI_list(List * data, double * vector_dbl, int* vector_int, int dim)

cdef extern from "brwre_black_2D.c":

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


# HAVE TO CHANGE THE TWO VALUES BELOW ALSO IN C FILES!
# Number of particles at the beginning.
cdef int INI_PART

INI_PART = 15000

# Size of the box:
cdef int CENTER
cdef int DIM_BOX

DIM_BOX = 150
CENTER  = 75

# Number of random times we generate at once:
cdef int DIM_RANDOM

DIM_RANDOM = 100000

# SPATIAL_NOISE is the (random) potential the particles move in.
cdef double SPATIAL_NOISE[150][150]
# We introduce an additional intensity constant, otherwise the effects are too small.
intensity = 5
# This is the true macroscopic noise.

noise_macro = np.random.normal(size = (DIM_BOX, DIM_BOX))*DIM_BOX*intensity
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

cdef double initial_dbl[15000]
cdef int initial_int[15000][3]

initial_times  = np.random.exponential(size = (INI_PART))
initial_choice = np.random.binomial(2, 0.5, size= (INI_PART))

for i in range(INI_PART):
	initial_dbl[i] = initial_times[i]
	initial_int[i][0] = CENTER
	initial_int[i][1] = CENTER
	initial_int[i][2] = initial_choice[i]

INI_list(ini_ptr, &initial_dbl[0], &initial_int[0][0], INI_PART)

# Describes how much time we let pass for every frame.
time_step = 0.0003
aim = time_step*(DIM_BOX**2)

# So that now we can initialize the BRWRE sample:
INI_brwre(particle, &SPATIAL_NOISE[0][0], &TIME_NOISE[0], ini_ptr, aim)

# Parameters for the sizes of the picture

B_NUM_EXTRA = 100
B_NUM_averaging = 30

# Compute the spatial subdivision.
# And start the brwre_histogram.

cdef LS_element* temp_list

cur_dim = particle[0].data[0].dim
temp_list = particle[0].data[0].First
python_loc = np.zeros(shape = (cur_dim, 2))
for i in range(cur_dim):
	python_loc[i,0] = temp_list[0].LS_value_int[0]
	python_loc[i,1] = temp_list[0].LS_value_int[1]
	temp_list = temp_list[0].LS_next

brwre_histo, space_histo_x, space_histo_y = np.histogram2d(python_loc[:,0], python_loc[:,1], bins = (B_NUM_EXTRA, B_NUM_EXTRA), \
	range = [[0, DIM_BOX],[0, DIM_BOX]])

avrg_histo, avrg_histo_x, avrg_histo_y = np.histogram2d(python_loc[:,0], python_loc[:,1], bins = (B_NUM_averaging, B_NUM_averaging), \
	range = [[0, DIM_BOX],[0, DIM_BOX]])

space_pts = len(space_histo_x)-1

space_averaging = len(avrg_histo_x)-1

noise_avrg = np.zeros(shape = (space_averaging, space_averaging))
for i in range(space_averaging):
	for j in range(space_averaging):
		for m in xrange( int(avrg_histo_x[i]), int(avrg_histo_x[i+1]) ):
			for n in xrange( int(avrg_histo_y[j]), int(avrg_histo_y[j+1]) ):
				noise_avrg[i, j] += noise_macro[m, n]

# Below, multiply with B_NUM is the average in space.
# Every box is macroscopically of size 1/B_NUM
# Divide with DIM_BOX is because of the Riemann integral:
# \int \xi^n(x)f(x) = \sum_x n^{-2} \xi^n(x) f(x)

noise_avrg = noise_avrg*(B_NUM_averaging/DIM_BOX)**2

# We define the animation function:
def animate(i):

	global cur_dim, aim, TIME_NOISE, brwre_histo

	# Real time is:
	ani_time = i*time_step
	
	# Redefine the plot for the particles
	im1.set_data(brwre_histo*(B_NUM_EXTRA/DIM_BOX)**2 )

	# Set the new time
	time_text.set_text("Time = {:2.3f}, Num prtcls: {}, Rho = {}".format(ani_time, cur_dim, 1.0))
	
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
	python_loc = np.zeros(shape = (cur_dim, 2))
	for i in range(cur_dim):
		python_loc[i,0] = temp_list[0].LS_value_int[0]
		python_loc[i,1] = temp_list[0].LS_value_int[1]
		temp_list = temp_list[0].LS_next

	brwre_histo, space_histo_x, space_histo_y = np.histogram2d(python_loc[:,0], python_loc[:,1], bins = (B_NUM_EXTRA, B_NUM_EXTRA), \
		range = [[0, DIM_BOX],[0, DIM_BOX]])

	return [im1,] + [time_text,]

# We set up the picture
fig       = plt.figure(figsize = (19, 8))
ax1       = fig.add_subplot(1,2,1)
ax2       = fig.add_subplot(1,2,2)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
time_text = ax1.text(0.05, 0.95,'',horizontalalignment='left',verticalalignment='top', transform=ax1.transAxes, color = 'white')
# Picture for the particle system
im1       = ax1.imshow(brwre_histo*(B_NUM_EXTRA/DIM_BOX)**2, interpolation='none', origin='low', vmin = 0, vmax = 0.034*intensity*B_NUM_EXTRA,  \
	extent=[space_histo_x[0]/DIM_BOX, space_histo_x[-1]/DIM_BOX, space_histo_y[0]/DIM_BOX, space_histo_y[-1]/DIM_BOX], cmap = plt.get_cmap('jet'))


# Picture for the averaged noise
im2       = ax2.imshow(noise_avrg, interpolation='none', origin='low', vmin = -2*intensity*B_NUM_averaging, vmax = 2*intensity*B_NUM_averaging,  \
	extent=[avrg_histo_x[0]/DIM_BOX, avrg_histo_x[-1]/DIM_BOX, avrg_histo_y[0]/DIM_BOX, avrg_histo_y[-1]/DIM_BOX], cmap = plt.get_cmap('plasma'))
#ax1.add_im(im1)
#ax2.add_im(im2)

#colmap    = plt.get_cmap('plasma') OR 'hot'

plt.title("2D BRWRE") 

# We let the animation go.
ani       = animation.FuncAnimation(fig, animate, frames=400, interval = 70, blit = True)
ani.save(filename = 'brwre_2D.mp4', extra_args=['-vcodec', 'libx264'], bitrate = 16000)

#END


# This is to check that the particle system works as required.

#cdef extern from "my_2D_list.c":
#	void PRINT_list_int(List* data, int row)
#	void PRINT_list_dbl(List* data, int row)
#
#cdef extern from "brwre_black_2D.c":
#	void DO_move(Brwre* particle)
#
#print(aim)
#particle[0].aim = 2.0
#
#for _ in xrange(1,20):
#	PRINT_list_dbl(particle[0].data, 0);
#	PRINT_list_int(particle[0].data, 0);
#	PRINT_list_int(particle[0].data, 1);
#	PRINT_list_int(particle[0].data, 2);
#	print(" \n Error number: {} and count = {} and END_reached = {} \n", particle[0].ERROR_NUMBER,
#			particle[0].count, particle[0].END_reached);
#	print("________________________________________\n");
#
#	DO_move(particle);

