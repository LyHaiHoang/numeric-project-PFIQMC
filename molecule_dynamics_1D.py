import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# https://compphys.quantumtinkerer.tudelft.nl/proj1-moldyn-description/
# http://www.nriv.free.fr/sciences/rapports/expose/jones1.pdf

# ==============================
# Initial parameters
# ==============================
N           = 50            # number of atoms
M           = 57*10e-3          # molar mass (kg/mol)
epsilon     = 400               # cohesion energy (K)
d           = 4*10e-10          # interatomic distance (m)
kb          = 1.380649e-23      # Boltzmann constant (J/K)
NA          = 6.02214076e23     # Avogadro number (1/mol)
nsteps      = 1000              # number of simulation steps
m           = M/NA              # mass of one atom (kg)

# 1. Calculate characteristic time

t_caract = d * np.sqrt(M/(epsilon*kb*NA)) 
print(f"Characteristic time: {t_caract:.10e} s")

dt = t_caract / 100             
print(f"Time step: {dt:.10e} s")

# 2. Microcanonic ensemble simulation (NVE)

def energy_itr(r1,r2,sigma,epsilon):
    # Lennard-Jones potential energy between two atoms
    r = np.linalg.norm(r1-r2)
    return epsilon*((sigma/r)**12 - 2*(sigma/r)**6)

def energy_tot(positions,sigma,epsilon):
    # Total potential energy of the system
    U = 0
    for i in range(N-1):
        for j in [i-1,i+1]:
            print("energy between atoms", i, j, ":", energy_itr(positions[i],positions[j],sigma,epsilon))
            U += energy_itr(positions[i],positions[j],sigma,epsilon)
    return U

def force_itr(r1,r2,sigma,epsilon):
    # Lennard-Jones force between two atoms
    r = np.linalg.norm(r1-r2)
    f = 12*epsilon*((sigma**12/r**13) - (sigma**6/r**7))
    return f

def force_local(r1,r2,r3,sigma,epsilon):
    # Force on atom 2 due to atoms 1 and 3
    f12 = force_itr(r1,r2,sigma,epsilon)
    f32 = force_itr(r3,r2,sigma,epsilon)
    f_local = f12 - f32
    return f_local

def acceleration(r1,r2,sigma,epsilon,m):
    # Acceleration of atom 1 due to atom 2
    f = force_itr(r1,r2,sigma,epsilon)
    a = f / m
    return a
# a. Verlet velocity algorithm

# Initilal conditions

def init_positions(N,d):
    # Intial positions of N atoms in a 1D chain
    x0 = 0.0
    x = np.zeros(N)            
    for i in range(N):
        x[i] = x0 + i*d
    return x.reshape((N,1))

def init_velocities(N,T):
    # Initial velocities of N atoms at temperature T
    v_theo = np.sqrt(kb*T/M)   
    v = np.random.normal(loc=0.0, scale=v_theo, size=(N,1))
    v_mean = v.mean(axis=0, keepdims=True)
    # print("Mean velocity:", v_mean)
    v = v - v_mean

    v2_mean = np.mean(v**2)
    if v2_mean > 0:
        scale = np.sqrt((kb*T/M) / v2_mean)
        v = v * scale
    return v

def force_global(positions,sigma,epsilon):
    # Global forces on all atoms
    N = positions.shape[0]
    forces = np.zeros_like(positions)
    for i in range(N):
        forces[i]=force_local(positions[i-1],positions[i],positions[(i+1)%N],sigma,epsilon)
    return forces

def compute_velocities(positions, velocities, dt, sigma, epsilon):
    N = positions.shape[0]
    forces = np.zeros_like(positions)
    for i in range(N):
        for j in range(N):
            if i != j:
                r_ij = positions[i] - positions[j]
                f_ij = force_itr(positions[i], positions[j], sigma, epsilon) # only for the nearest neighbor
                forces[i] += f_ij * (r_ij / np.linalg.norm(r_ij))  # direction of the force

    # Update velocities and positions
    velocities += (forces / M) * dt
    positions += velocities * dt
    return positions, velocities


x = init_positions(N,d)
v = init_velocities(N,epsilon)
f = force_global(x, d, epsilon)
e = energy_tot(x, d, epsilon)

print("Initial positions:", x)
print("Initial velocities:", v)
print("Mean initial velocity:", v.mean(axis=0, keepdims=True))
print("Forces:", f)
print("Initial total potential energy:", e)







