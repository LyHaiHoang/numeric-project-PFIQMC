import numpy as np
import matplotlib.pyplot as plt

# Parameters (you can adjust)
mode = 1    # 1: all up, 2: all down, 3: random, 4: half up half down
kb = 1.0
J = 1.0
H_ext = 0.0
L = 30       # lattice linear size
N = L * L

print("Particle number:", N)

def init_spins(L, mode, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    if mode == 1:
        return np.ones((L, L), dtype=int)
    elif mode == 2:
        return -np.ones((L, L), dtype=int)
    elif mode == 3:
        return rng.choice([1, -1], size=(L, L))
    elif mode == 4:
        n = L * L
        half = n // 2
        flat = np.array([1]*half + [-1]*(n-half), dtype=int)
        rng.shuffle(flat)
        return flat.reshape(L, L)
    else:
        raise ValueError("mode must be 1..4")

def total_energy(spins, J=1.0, H=0.0):
    """Total energy without double counting (use right+down neighbors)."""
    L = spins.shape[0]
    E = 0.0
    for i in range(L):
        for j in range(L):
            s = spins[i, j]
            right = spins[i, (j+1) % L]
            down  = spins[(i+1) % L, j]
            E += -J * s * (right + down)
            E += -H * s
    return E

def magnetization(spins):
    return spins.sum() / (spins.size)   # per spin

def metropolis_sweep(spin_tot, beta, J, H=0.0):
    """One Metropolis sweep (N attempted flips). In-place update."""
    L = spin_tot.shape[0]
    Nloc = L * L
    for _ in range(Nloc):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        s = spin_tot[i, j]
        nb = (spin_tot[(i-1) % L, j] + spin_tot[(i+1) % L, j] +
              spin_tot[i, (j-1) % L] + spin_tot[i, (j+1) % L])
        delta_E = 2 * J * s * nb + 2 * H * s   # include external field H if nonzero
        if delta_E <= 0 or np.random.rand() < np.exp(-beta * delta_E):
            spin_tot[i, j] = -s
    return spin_tot

def run_metropolis_single_T(L, T, mode, J, H, kb, n_eq_sweeps=500, n_meas_sweeps=1000, rng_seed=None):
    """Run Metropolis at single temperature T and return magnetization time series (per spin)."""
    rng = np.random.default_rng(rng_seed)
    beta = 1.0 / (kb * T)
    spins = init_spins(L, mode, rng=rng)
    # equilibration
    for _ in range(n_eq_sweeps):
        metropolis_sweep(spins, beta, J, H)
    # measurement: collect m after each sweep
    m_vals = []
    for _ in range(n_meas_sweeps):
        metropolis_sweep(spins, beta, J, H)
        m_vals.append((magnetization(spins)))   # store |m| per spin
    return np.array(m_vals)

if __name__ == "__main__":
    # initial example: print initial config for small L
    L_print = 30
    spins0 = init_spins(min(L, L_print), mode=mode)
    for i in range(spins0.shape[0]):
        print(" ".join(f"{x:2d}" for x in spins0[i,:]))
    print("Initial |m| (per spin):", abs(magnetization(spins0)))

    # critical temperature (2D Ising, exact)
    Tc = 2 * J / (np.log(1 + np.sqrt(2)))
    print("Tc (2D Ising):", Tc)

    # sweep temperatures and compute average |m| (after equilibration)
    temperatures = np.linspace(1.0, 4.0, 100)
    m_vals_avg = []
    # Use moderate runs by default; increase for better accuracy
    n_eq_sweeps = 300
    n_meas_sweeps = 300
    for Tval in temperatures:
        m_time_series = run_metropolis_single_T(L, Tval, mode, J, H_ext, kb,
                                                n_eq_sweeps=n_eq_sweeps,
                                                n_meas_sweeps=n_meas_sweeps,
                                                rng_seed=None)
        # take average of |m| over measurement sweeps
        m_vals_avg.append(m_time_series.mean())

    plt.figure(figsize=(6,4))
    plt.plot(temperatures, m_vals_avg, 'o-', label=r'$\langle |m| \rangle$')
    plt.axvline(Tc, color='r', linestyle='--', label=f'Tc ~ {Tc:.3f}')
    plt.xlabel('Temperature T')
    plt.ylabel('Magnetization |m|')
    plt.title(f'2D Ising Metropolis (L={L}, mode={mode})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
