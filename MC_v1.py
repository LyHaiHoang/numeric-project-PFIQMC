# This code is used to calculate the ferro-paramagnetics phase transiion of 2D Ising model by using the Monte Carlo method
import numpy as np
import matplotlib.pyplot as plt


# Initialization
mode = 3    # 1: all up, 2: all down, 3: random - spin configuration, 4: half up half down
kb = 1.0    # Boltzmann constant
T = 1.0     # Temperature
beta = 1/(kb*T)
J = 1.0     # Interaction strength
H = 0.0     # External magnetic field
L = 6       # Lattice size
N = L * L   # Number of spins
print("Particle number:", N)




def init_spins(L, mode):
    if mode == 1:
        spins = np.ones((L, L), dtype=int)
    elif mode == 2:
        spins = -np.ones((L, L), dtype=int)
    elif mode == 3:
        spins = np.where(np.random.rand(L, L) < 0.5, 1, -1)
    elif mode == 4:
        n = L * L
        half = n // 2
        spins_flat = np.array([1]*half + [-1]*(n - half), dtype=int)
        np.random.shuffle(spins_flat)
        spins = spins_flat.reshape(L, L)
    else:
        raise ValueError("Please choose mode 1, 2, 3, or 4.")

    # Thêm 1 hàng + 1 cột để áp dụng periodic boundary condition
    m = np.zeros((L+1, L+1), dtype=int)
    m[:L, :L] = spins

    # Điều kiện biên tuần hoàn
    m[L, :L] = m[0, :L]     # hàng cuối = hàng đầu
    m[:L, L] = m[:L, 0]     # cột cuối = cột đầu
    m[L, L]   = m[0, 0]     # góc cuối = góc đầu

    return m


def hamiltonian(spin_tot, J, H):
    H = H1 = H2 = 0.0
    # Calculate hamiltonian interaction between spins
    for i in range(L):
        for j in range(L):
            H1 += -J * (spin_tot[i,j]*(spin_tot[i,(j-1)] + spin_tot[(i-1),j] + spin_tot[i,(j+1)] + spin_tot[(i+1),j]))
    # Calculate hamiltonian interaction with magnetic field H
    for i in range(L):
        for j in range(L):
            H2 += -H * spin_tot[i,j]
    return H1/4 + H2

def susceptibility(m2, m, T, N):
    x = (N**2)*(m2 - m**2)/(kb*T)
    return x

def energy_moy(H_tot, N):
    e = H_tot / N
    return e

def specific_heat(e2, e, T, N):
    Cv = (N**2)*(e2 - e**2)/(kb*T**2)
    return Cv

def metropolis_sweep(spin_tot, beta, J):
    for _ in range(N):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        delta_E = 2 * J * spin_tot[i,j] * (spin_tot[i,(j-1)%L] + spin_tot[(i-1)%L,j] + spin_tot[i,(j+1)%L] + spin_tot[(i+1)%L,j])
        if delta_E < 0 or np.random.rand() < np.exp(-beta * delta_E):
            spin_tot[i,j] *= -1
    return spin_tot

def aimantation(spin_tot):
    m = np.sum(spin_tot) / N
    return m

def plot_mT(L, J, H, T_min, T_max, nT, N):
    T_min = 1.0
    T_max = 4.0
    nT = 20
    Ts = np.linspace(T_min, T_max, nT)
    mvals = []
    lattice = init_spins(L,4)

    for T in Ts:
                
        acc = 0.0
        for _ in range(N):
            metropolis_sweep(lattice, beta, J)
            acc = aimantation(lattice)
        mvals.append(acc)
    return Ts, np.array(mvals)

if __name__ == "__main__":

    # 1.1 Initial spin configuration
    spin_tot = init_spins(N, mode)
    # Compute corrlation function
    for i in range(L):
        for j in range(L):
            print("\t",spin_tot[i,j], end=' ')
        print()
    m = aimantation(spin_tot)
    m2 = np.sum(spin_tot**2)/N
    print("Initial aimantation:", m)

    # 1.2. Hamiltonian calculation
    H = 0.0     # External magnetic field
    H_tot = hamiltonian(spin_tot, J, H)
    print("Initial Hamiltonian:", H_tot)

    # 1.3. Susceptibility calculation
    susceptibility = susceptibility(m2, m, T, N)
    print("Initial susceptibility:", susceptibility)

    # 1.4. Energy per spin calculation
    e = energy_moy(H_tot, N)
    e2 = energy_moy(H_tot**2, N)
    print("Initial energy per spin:", e)

    # 1.5. Specific heat calculation
    Cv = specific_heat(e2, e, T, N)
    print("Initial specific heat:", Cv)

    # 1.6 Compute critical temperature
    Tc = 2*J / (np.log(1 + np.sqrt(2)))
    print("Critical temperature:", Tc)

    #2.0 Monte Carlo simulation
    
    Ts, mvals = plot_mT(L, J, H, T_min=1.0, T_max=4.0, nT=20, N=N)
    Tc = 2.0 / np.log(1+np.sqrt(2))
    plt.plot(Ts, mvals, 'o-', label='|m| (Metropolis)')
    plt.axvline(Tc, color='r', ls='--', label=f'Tc~{Tc:.3f}')
    plt.xlabel('T')
    plt.ylabel('magnetization per spin |m|')
    plt.legend()
    plt.show()