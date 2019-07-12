import numpy as np

# To time stuff we import the measure decorator from my_timeit.py
from my_timeit import measure

# Create a grid
grid = np.arange(start = 0.0, stop = 10.0, step = 0.1)
grid_list = [grid]*2
mesh = np.array(np.meshgrid(*grid_list)).T

# Define a function (ideally 2D input)
def myfunc(input):
	return np.abs( 0.2* np.cos(np.dot(input, input)))

myfunc_vec = np.vectorize(myfunc, signature = '(n)->()')

# Vectorize the function
@measure
def func_vec(input):
	return myfunc_vec(input)

# Extend it via loops
@measure
def func_loop(input):
	result = np.zeros(shape = input.shape[:-1])
	for i in range(result.shape[0]):
		for j in range(result.shape[1]):
			result[i,j] = myfunc(input[i,j,:])
	return result

func_vec(mesh)
func_loop(mesh)

#RESULT: With signatures vectorize is outperformed by a Python loop!!

# The same in 1D
grid = np.arange(start = 0.0, stop = 10000.0, step = 0.1)


# Define a function (ideally 2D input)
def myfunc(input):
	return np.abs( 0.2* np.cos(np.dot(input, input)))

myfunc_vec = np.vectorize(myfunc)

# Vectorize the function
@measure
def func_vec(input):
	return myfunc_vec(input)

# Extend it via loops
@measure
def func_loop(input):
	result = np.zeros(shape = input.shape[:])
	for i in range(result.shape[0]):
		result[i] = myfunc(input[i])
	return result

func_vec(grid)
func_loop(grid)

#RESULT: WIthout signatures this is the same as a python loop!!

# we try NUMBA:
from numba import vectorize, float64

@measure
@vectorize([float64(float64)])
def func_numba(input):
	return np.abs( 0.2* np.cos(np.dot(input, input)))

func_numba(grid)

# We try with Cython:

import cython

@measure
@cython.cfunc
@cython.returns(cython.int)
@cython.locals(a=cython.int)
def func_cython(a):
	for count in range(1000000):
		a = a+1
	return a

@measure
def python_loop(a):
	for count in range(1000000):
		a = a + 1
	return a

func_cython(1)
python_loop(1)