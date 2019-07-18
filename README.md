# Branching random walks in a random environment

Fix an environment, that is a collection <img src="/tex/834279909e8addaa426c5634bd064227.svg?invert_in_darkmode&sanitize=true" align=middle width=84.72253679999999pt height=27.91243950000002pt/> of i.i.d. random variables.

We simulate a system of particles (in dimension <img src="/tex/13ddb09bb6adaa52ddb96197a18570f6.svg?invert_in_darkmode&sanitize=true" align=middle width=54.217896149999994pt height=22.831056599999986pt/>) which, independently of each other:

- Perform a random walk (in 1 or 2 dimensions)

- Branch with a rate proportional to the value of a random (white) potential
at the location of the particle. That is, if <img src="/tex/a918cf04cd0ac7535e7626be634cfb9e.svg?invert_in_darkmode&sanitize=true" align=middle width=18.58454399999999pt height=22.465723500000017pt/> is the position of the
particle at time <img src="/tex/4f4f4e395762a3af4575de74c019ebb5.svg?invert_in_darkmode&sanitize=true" align=middle width=5.936097749999991pt height=20.221802699999984pt/>:

  - The probability of producing a new offspring is proportional to <img src="/tex/b6788ceda860c2586eb4c6659d7072ce.svg?invert_in_darkmode&sanitize=true" align=middle width=102.6971847pt height=24.65753399999998pt/>.
  - The probability of dying is proportional to <img src="/tex/44337222f71b71e7a636d5fe53545522.svg?invert_in_darkmode&sanitize=true" align=middle width=115.0259088pt height=24.65753399999998pt/>.

The simulation of this system is straightforward. The heart is implemented in C
to be able to simulate large quantities of particles. The C code is imported in
Python with Cython, where pictures and short movies can be produced. Both
dimensions work essentially in the same way.

__Description of the programs (2D case only)__

- The particle system at fixed time is a C list (these are defined in
  my_2D_list.c / my_2D_list.h)

- The particle system is run in brwre_black_2D.c, which takes as an input the
  random environment and a set of externally generated pseudo-random variables.
  The particle system runs until there are no more unused random variables.

- The C code is embedded in Python in brwre_2D.pyx (which can be run with
  setup.py - see instructions in file). Here the output is a movie. The
  particle system is run at under diffusive scaling.

Motivation for this simulation was the study of the scaling limit of the
particle system:

- https://arxiv.org/abs/1905.05825

- https://arxiv.org/abs/1906.11054

Small number of particles:

![alt text](brwre_1.gif)

Large number of particles:

![alt text](brwre_2.gif)
