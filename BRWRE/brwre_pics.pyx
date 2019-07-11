# We model BRWRE with time dependend environment

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors
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


# HAVE TO CHANGE THE TWO VALUES BELOW ALSO IN C FILES!
# Number of particles at the beginning.
cdef int INI_PART

INI_PART = 15000

# Size of the box:
cdef int CENTER
cdef int DIM_BOX

DIM_BOX = 15000
CENTER  = 7500

# Number of random times we generate at once:
cdef int DIM_RANDOM

DIM_RANDOM = 200000

# SPATIAL_NOISE is the (random) potential the particles move in.
cdef double SPATIAL_NOISE[15000]

SPATIAL_NOISE = (np.random.normal(size = (DIM_BOX))*np.sqrt(DIM_BOX))/DIM_BOX**2
#SPATIAL_NOISE = np.zeros(shape = (DIM_BOX))

# TIME_NOISE is the number of random times we generate.
cdef double TIME_NOISE[200000]
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
cdef int initial_int[15000][2]

initial_times  = np.sort(np.random.exponential(size = (INI_PART)))
initial_choice = np.random.binomial(1, 0.5, size= (INI_PART))

for i in range(INI_PART):
	initial_dbl[i] = initial_times[i]
	initial_int[i][0] = CENTER
	initial_int[i][1] = initial_choice[i]

INI_list(ini_ptr, &initial_dbl[0], &initial_int[0][0], INI_PART)


# Describes how much time we let pass for every frame.
time_step = 0.005
aim = time_step*(DIM_BOX**2)

# So that now we can initialize the BRWRE sample:
INI_brwre(particle, &SPATIAL_NOISE[0], &TIME_NOISE[0], ini_ptr, aim)

error = 2
count_steps = 0

print( "Have to get to {:2.3f} \n".format(aim))
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
	
	sys.stdout.write("\r Step = {}. This step took me {:2.3f} s. Percentual on total: {:1.5f}. Elapsed total: {:4.2f}. Hrs to end:{:4.2f}. Particles {:d}".format( \
		count_steps, time.time()-start_time, particle[0].cur_particle/particle[0].data[0].dim, \
		time.time()-start_time_out, (time.time()-start_time_out)*(particle[0].data[0].dim- \
		particle[0].cur_particle)/(particle[0].cur_particle*3600), particle[0].data[0].dim ) )
	
	count_steps += 1	

print("\n \n ")
sys.stdout.flush()
sys.stdout.write("Finished picture in {:2.3f} seconds".format(time.time()-start_time_out))
print("\n \n")

# Pass the values to Python
cdef LS_element* temp_list

cur_dim = particle[0].data[0].dim
temp_list = particle[0].data[0].First
python_loc = np.zeros(shape = (cur_dim))
for i in range(cur_dim):
	python_loc[i] = temp_list[0].LS_value_int[0]
	temp_list = temp_list[0].LS_next

# Parameters for the sizes of the picture
#B_SIZE  = 45
#space   = np.arange(0, 1+ 0.01, 1/B_SIZE)
B_NUM = 60

# Compute the histogram
brwre_histo, brwre_bins = np.histogram(python_loc, bins = B_NUM, range = (0, DIM_BOX), density = False)

# We set up the picture
# Real axes length

#lines,    = ax.plot(brwre_bins[:-1]/DIM_BOX, brwre_histo*B_SIZE/DIM_BOX, lw = 2)
#lines,    = ax.plot(brwre_bins[:-1]/DIM_BOX, np.exp(), lw = 2)


# Add the plot
#lines.set_data(brwre_bins[:-1]/DIM_BOX, brwre_histo*B_SIZE/DIM_BOX)

	
# Add the data of the pictures:

fig       = plt.figure()
ax        = plt.axes(xlim=(0, 1), ylim = (0, 5))

ax.plot(brwre_bins[:-1]/DIM_BOX, brwre_histo*B_NUM/DIM_BOX, lw = 2)
ax.plot(brwre_bins[:-1]/DIM_BOX, np.exp(-((brwre_bins[:-1]-CENTER)/DIM_BOX)**2/(4*time_step))/np.sqrt(4*np.pi*time_step), lw = 2)

plt.title("Branching RW in random environment") 
time_text = ax.text(0.05, 0.95,'',horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
time_text.set_text("Time = {:2.3f}, Num prtcls: {}, Rho = {}".format(time_step, cur_dim, 1.0) )

plt.savefig("Brwre_with_potential.png")
