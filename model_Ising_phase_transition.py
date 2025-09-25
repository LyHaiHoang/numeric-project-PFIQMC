# This code is used to calculate the ferro-paramagnetics phase transiion of 2D Ising model by using the Monte Carlo method
import numpy as np
import matplotlib.pyplot as plt


# Initialization
mode = 3    # 1: all up, 2: all down, 3: random - spin configuration
kb = 1.0    # Boltzmann constant
J = 1.0     # Interaction strength
L = 1000    # Lattice size
N = L * L   # Number of spins
print("Particle number:", N)


def init_spins(N, mode):
    if mode == 1:
        return np.ones((L,L), dtype=int)
    elif mode == 2:
        return -np.ones((L,L), dtype=int)
    elif mode == 3:
        return np.where(np.random.rand(L,L) < 0.5, 1, -1)
    else:
        raise ValueError("Plese choose mode 1, 2, or 3.")
    
    # surface periodic condition
    for j in range(L):
        m[L,j] = m[1,j]
    for i in range(L):
        m[i,L] = m[i,1]


if __name__ == "__main__":
    m = init_spins(N, mode)
    print("Initial aimantation:", np.sum(m)/N)
    print(m[1,20])
    print(m[999,20])