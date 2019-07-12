import numpy as np

cdef int limit = 1000000000
cdef int count
cdef int result
cdef int loop(int a):
	for count in range(limit):
		a = a+1
	return a
# Decorators do not work with cdef functions (I think)
from time import time, sleep
start  = time()
sleep(2)

result = loop(1)
print(result)

end_   = time() - start
print(f"Total execution time: {end_ if end_ > 0 else 0} s")

from my_timeit import measure

@measure
def python_loop(a):
	for _ in range(1000000000):
		a = a + 1
	return a

result_p = python_loop(1)

print(result, result_p)

# The firt loop took 376 ms
# The second loop 19746 ms
# So the first used 1.9% of the time of the second.
