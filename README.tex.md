# Branching random walks in a random environment

Fix an environment, that is a collection $\xi(x), x \in \mathbb{Z}^d$ of i.i.d. random variables.

We simulate a system of particles (in dimension $d=1, 2$) which, independently of each other:

- Perform a random walk (in 1 or 2 dimensions)

- Branch with a rate proportional to the value of a random (white) potential
at the location of the particle. That is, if $X_t$ is the position of the
particle at time $t$:

  - The probability of producing a new offspring is proportional to $\max \{ \xi(X_t), 0 \}$.
  - The probability of dying is proportional to $-\min \{ \xi(X_t), 0 \}$


