import numpy as np
import matplotlib.pyplot as plt


# === Paramètres ===
kb = 1.0           # Constante de Boltzmann (fixée à 1 pour simplifier)
T = 1.5            # Température initiale
J = 1.0            # Constante d’interaction entre spins (échange)
L = 50             # Taille du réseau (L x L)
N = L * L          # Nombre total de spins
H = 0.0
n_mcs   = 500     # Nombre d’itérations Monte Carlo par valeur de H


# === Initialisation des spins ===
def spin_init(L, mode=1):
    if mode == 1:
        return np.ones((L, L), dtype=int)
    else:
        return -np.ones((L, L), dtype=int)


# === Variation d’énergie ΔE lors d’un retournement de spin ===
def delta_E(spins, L, i, j, H, J):
    s = spins[i, j]
    # Somme des 1er voisins proches (haut, bas, gauche, droite)
    nn = (spins[(i+1) % L, j] + spins[(i-1) % L, j] + spins[i, (j+1) % L] + spins[i, (j-1) % L])
    return 2 * s * (J * nn + H)


# === Algorithme de Metropolis ===
def metropolis(spins, L, H, J, T):
    beta = 1.0 / (kb * T)
    for _ in range(L * L):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        dE = delta_E(spins, L, i, j, H, J)
        if dE <= 0 or np.random.rand() < np.exp(-beta * dE):
            spins[i, j] *= -1  # Retourne le spin
    return spins


# === Magnétisation ===
def aimantation(spins, N):
    return np.sum(spins) / N 

# === Balayage en température ===
T_start, T_end, dT = 1.0, 4.0, 0.02   
T_up   = np.arange(T_start, T_end + dT, dT)
T_down = np.arange(T_end, T_start - dT, -dT)
T_vals = np.concatenate([T_up, T_down])

# === Balayage en champ magnétique ===
H_start, H_end, dH = -3.0, 3.0, 0.2   
H_up   = np.arange(H_start, H_end + dH, dH)   
H_down = np.arange(H_end, H_start - dH, -dH) 
H_vals = np.concatenate([H_up, H_down])


spins = spin_init(L, mode=1)  # Tous les spins vers le haut
print("Initialisation des spins", spins) 
mT_vals = []
mH_vals = []


choix = 1

if choix == 1:
    # Variation de la température
    for T in T_vals:
        print(f"Température: {T:.2f}")
        # On laisse évoluer le système vers l’équilibre
        for _ in range(n_mcs):  
            spins = metropolis(spins, L, H=0.0, J=J, T=T)
        # On mesure la magnétisation
        mT_vals.append(aimantation(spins, N))

    filename_mT = f"mT_L{L}_step{n_mcs}_H{H}.txt"
    data_mT = np.column_stack([T_vals, mT_vals])
    np.savetxt(filename_mT, data_mT, fmt="%.6f")
elif choix == 2:
    # === Boucle d’hystérésis (variation de H à T fixe) ===
    T = 2.5  # Température fixée pour la boucle
    for H in H_vals:
        print(f"Champ magnétique: {H:.2f}")
        # On laisse évoluer le système vers l’équilibre
        for _ in range(n_mcs):
            spins = metropolis(spins, L, H, J, T)
        # On enregistre la magnétisation
        mH_vals.append(aimantation(spins, N))
    filename_mH = f"mH_L{L}_step{n_mcs}_T{T}.txt"
    data_mH = np.column_stack([H_vals, mH_vals])
    np.savetxt(filename_mH, data_mH, fmt="%.6f")

elif choix == 3:
    # === Calcul de la chaleur latente magnétique selon Clausius–Clapeyron ===
    print("=== Calcul de Lm(T) selon Clausius–Clapeyron ===")

    # Paramètres du balayage
    H_start, H_end, dH = -3.0, 3.0, 0.2
    H_vals = np.concatenate([
        np.arange(H_start, H_end + dH, dH),
        np.arange(H_end, H_start - dH, -dH)
    ])

    T_start, T_end, dT = 1.0, 4.0, 0.2
    T_vals = np.arange(T_start, T_end + dT, dT)

    # Listes de résultats
    Hc_vals, mplus_vals, mminus_vals = [], [], []

    for T in T_vals:
        spins = spin_init(L, mode=1)
        mH = []

        # --- Boucle d'hystérésis à chaque T ---
        for H in H_vals:
            for _ in range(n_mcs):
                spins = metropolis(spins, L, H, J, T)
            mH.append(aimantation(spins, N))

        mH = np.array(mH)

        # --- Calcul des grandeurs caractéristiques ---
        # Aimantations rémanentes à H → 0⁺ et H → 0⁻
        idx_zero_pos = np.where(H_vals > 0)[0][0]
        idx_zero_neg = np.where(H_vals < 0)[0][-1]
        m_plus = mH[idx_zero_pos]
        m_minus = mH[idx_zero_neg]

        # Champ coercitif Hc (interpolation où M=0)
        sign_change_idx = np.where(np.diff(np.sign(mH)))[0]
        if len(sign_change_idx) > 0:
            i = sign_change_idx[0]
            Hc = H_vals[i] - mH[i] * (H_vals[i+1] - H_vals[i]) / (mH[i+1] - mH[i])
        else:
            Hc = np.nan

        # Sauvegarde temporaire
        Hc_vals.append(Hc)
        mplus_vals.append(m_plus)
        mminus_vals.append(m_minus)
        print(f"T={T:.2f} K → Hc={Hc:.3f}, m+={m_plus:.3f}, m-={m_minus:.3f}")

    # --- Conversion en arrays ---
    T_vals = np.array(T_vals)
    Hc_vals = np.array(Hc_vals)
    mplus_vals = np.array(mplus_vals)
    mminus_vals = np.array(mminus_vals)

    # --- Calcul de la dérivée dHc/dT ---
    dHc_dT = np.gradient(Hc_vals, T_vals)

    # --- Calcul de la chaleur latente Lm(T) ---
    Lm = T_vals * (mplus_vals - mminus_vals) * dHc_dT

    # --- Sauvegarde des résultats ---
    filename_LmT = f"LmT_L{L}_step{n_mcs}.txt"
    data_LmT = np.column_stack([T_vals, mplus_vals, mminus_vals, Hc_vals, dHc_dT, Lm])
    header = "T(K)\tm_plus\tm_minus\tHc(T)\tdHc/dT\tLm(T)"
    np.savetxt(filename_LmT, data_LmT, fmt="%.6f", header=header)
    print(f"\nRésultats sauvegardés dans '{filename_LmT}'")

  

    
