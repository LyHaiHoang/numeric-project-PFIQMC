"""
ising2d_mc.py
Monte Carlo (Metropolis) simulation of the 2D Ising model.
- Periodic boundary conditions
- Observables: energy per spin, magnetization per spin, specific heat, susceptibility
- Plots: E(T), |M|(T), C(T), chi(T) and snapshots of spin lattice at selected temperatures

Author: ChatGPT (example)
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # optional, for progress bar. Install with `pip install tqdm` if needed.

# -----------------------
# Utility functions
# -----------------------
def initial_state(L, state='random'):
    """Return LxL spin configuration array with values +-1.
       state: 'random' or 'up' or 'down'"""
    if state == 'random':
        return np.where(np.random.rand(L, L) < 0.5, 1, -1)
    elif state == 'up':
        return np.ones((L, L), dtype=int)
    elif state == 'down':
        return -np.ones((L, L), dtype=int)
    else:
        raise ValueError("state must be 'random','up' or 'down'")

def calc_energy(spins, J=1.0):
    """Total energy of configuration (no external field), using periodic BC.
       H = -J sum_<ij> s_i s_j. We will sum each neighbor pair once."""
    L = spins.shape[0]
    # interaction in x and y directions
    energy = 0.0
    # right neighbor
    energy -= J * np.sum(spins * np.roll(spins, -1, axis=1))
    # down neighbor
    energy -= J * np.sum(spins * np.roll(spins, -1, axis=0))
    return energy

def calc_magnetization(spins):
    """Total magnetization (sum of spins)."""
    return np.sum(spins)

# -----------------------
# Metropolis update
# -----------------------
def metropolis_step(spins, beta, J=1.0):
    """Perform one Monte Carlo sweep (L*L attempted flips) using Metropolis.
       Returns updated spins and total delta energy & delta magnetization (for bookkeeping)."""
    L = spins.shape[0]
    # choose L*L random sites (could be shuffled list)
    for _ in range(L*L):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        s = spins[i, j]
        # sum of neighbor spins (periodic)
        nb = spins[(i+1)%L, j] + spins[(i-1)%L, j] + spins[i, (j+1)%L] + spins[i, (j-1)%L]
        dE = 2 * J * s * nb   # energy change if flip s -> -s
        if dE <= 0 or np.random.rand() < np.exp(-beta * dE):
            spins[i, j] = -s
    return spins

# -----------------------
# Simulation routine
# -----------------------
def simulate(L=20, temps=None, J=1.0, h=0.0,
             n_eq_sweeps=500, n_meas_sweeps=2000, start_state='random',
             seed=None, snapshot_temps=None):
    """
    Run simulation for a list of temperatures.
    Returns dictionaries of observables (per spin): E, M, C, chi and snapshots.
    snapshot_temps: list of T at which spin snapshots will be returned
    """
    if seed is not None:
        np.random.seed(seed)

    if temps is None:
        temps = np.linspace(1.5, 3.5, 21)  # around critical ~2.269 for 2D Ising (kB=1, J=1)

    N = L * L
    results = {'T': [], 'E': [], 'Mabs': [], 'E_var': [], 'M_var': []}
    snapshots = {}

    for T in tqdm(temps, desc="Temperatures"):
        beta = 1.0 / T
        spins = initial_state(L, state=start_state)

        # Equilibration
        for _ in range(n_eq_sweeps):
            metropolis_step(spins, beta, J=J)

        # Measurement
        E_acc = 0.0
        E2_acc = 0.0
        M_acc = 0.0
        M2_acc = 0.0
        n_meas = n_meas_sweeps

        for _ in range(n_meas):
            metropolis_step(spins, beta, J=J)
            E = calc_energy(spins, J=J) + -h * calc_magnetization(spins)  # include field if needed
            M = calc_magnetization(spins)
            E_acc += E
            E2_acc += E*E
            M_acc += M
            M2_acc += M*M

        E_mean = E_acc / n_meas
        E2_mean = E2_acc / n_meas
        M_mean = M_acc / n_meas
        M2_mean = M2_acc / n_meas

        # per spin
        E_per_spin = E_mean / N
        Mabs_per_spin = np.abs(M_mean) / N  # average magnetization (absolute)
        # fluctuations -> heat capacity and susceptibility
        C = (E2_mean - E_mean**2) / (N * T**2)   # specific heat per spin (k_B=1)
        chi = (M2_mean - M_mean**2) / (N * T)    # susceptibility per spin (k_B=1)

        results['T'].append(T)
        results['E'].append(E_per_spin)
        results['Mabs'].append(Mabs_per_spin)
        results['E_var'].append(C)
        results['M_var'].append(chi)

        # store snapshot copy if requested
        if snapshot_temps is not None and (np.any(np.isclose(snapshot_temps, T))):
            snapshots[T] = np.copy(spins)

    return results, snapshots

# -----------------------
# Plotting helpers
# -----------------------
def plot_observables(results):
    T = np.array(results['T'])
    E = np.array(results['E'])
    M = np.array(results['Mabs'])
    C = np.array(results['E_var'])
    chi = np.array(results['M_var'])

    plt.figure()
    plt.plot(T, E, marker='o')
    plt.xlabel('T')
    plt.ylabel('Energy per spin')
    plt.title('E(T)')
    plt.grid(True)

    plt.figure()
    plt.plot(T, M, marker='o')
    plt.xlabel('T')
    plt.ylabel('|M| per spin')
    plt.title('|M|(T)')
    plt.grid(True)

    plt.figure()
    plt.plot(T, C, marker='o')
    plt.xlabel('T')
    plt.ylabel('Specific heat per spin C')
    plt.title('C(T)')
    plt.grid(True)

    plt.figure()
    plt.plot(T, chi, marker='o')
    plt.xlabel('T')
    plt.ylabel('Susceptibility per spin χ')
    plt.title('χ(T)')
    plt.grid(True)

    plt.show()

def show_snapshot(spins, title=None):
    plt.figure(figsize=(5,5))
    # imshow with spins mapped to 0/1 for visualization; do not set colors explicitly
    plt.imshow(spins, interpolation='nearest')
    plt.title(title if title is not None else 'spin config')
    plt.axis('off')
    plt.show()

# -----------------------
# Example run
# -----------------------
if __name__ == '__main__':
    # Parameters you can tweak
    L = 40
    temps = np.concatenate((
        np.linspace(1.0, 2.0, 6),
        np.linspace(2.05, 2.8, 16),
        np.linspace(3.0, 4.0, 5)
    ))
    n_eq_sweeps = 800        # equilibration sweeps (increase for better accuracy)
    n_meas_sweeps = 1200     # measurement sweeps (increase for smoother curves)
    seed = 12345

    # Temperatures at which to save snapshots
    snapshot_temps = [1.5, 2.269, 3.5]

    results, snapshots = simulate(L=L, temps=temps, J=1.0, h=0.0,
                                  n_eq_sweeps=n_eq_sweeps,
                                  n_meas_sweeps=n_meas_sweeps,
                                  start_state='random',
                                  seed=seed,
                                  snapshot_temps=snapshot_temps)

    plot_observables(results)

    # Show snapshots
    for T in sorted(snapshots.keys()):
        show_snapshot(snapshots[T], title=f"T = {T:.3f}")
