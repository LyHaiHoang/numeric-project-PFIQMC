import numpy as np
import matplotlib.pyplot as plt

# === Paramètres ===
kb = 1.0           # Constante de Boltzmann (fixée à 1 pour simplifier)
T = 1.5            # Température initiale
J = 1.0            # Constante d’interaction entre spins (échange)
L = 50             # Taille du réseau (L x L)
N = L * L          # Nombre total de spins
n_steps = 20       # Nombre de valeurs de champ externe H (balayage)
n_mcs   = 100       # Nombre d’itérations Monte Carlo par valeur de H


# === Initialisation des spins ===
def spin_init(L, mode=1):
    if mode == 1:
        return np.ones((L, L), dtype=int)
    else:
        return -np.ones((L, L), dtype=int)


# === Variation d’énergie ΔE lors d’un retournement de spin ===
def delta_E(spins, i, j, H, J):
    L = spins.shape[0]
    s = spins[i, j]
    # Somme des voisins proches (haut, bas, gauche, droite)
    nn = (spins[(i+1) % L, j] + spins[(i-1) % L, j] +
          spins[i, (j+1) % L] + spins[i, (j-1) % L])
    return 2 * s * (J * nn + H)


# === Algorithme de Metropolis ===
def metropolis(spins, H, J, T):
    L = spins.shape[0]
    beta = 1.0 / (kb * T)
    for _ in range(L * L):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        dE = delta_E(spins, i, j, H, J)
        if dE <= 0 or np.random.rand() < np.exp(-beta * dE):
            spins[i, j] *= -1  # Retourne le spin
    return spins


# === Magnétisation ===
def aimantation(spins):
    return np.sum(spins) / spins.size


# === Balayage en température ===
H_vals = np.concatenate([np.linspace(-3, 3, n_steps), np.linspace(3, -3, n_steps)])
T_vals = np.concatenate([np.linspace(1.0, 4.0, n_steps), np.linspace(4.0, 1.0, n_steps)])

spins = spin_init(L, mode=1)  # Tous les spins vers le haut
print("Initialisation des spins", spins)
mT_vals = []
mH_vals = []

# Variation de la température
for T in T_vals:
    # On laisse évoluer le système vers l’équilibre
    for _ in range(n_steps):  
        spins = metropolis(spins, H=0.0, J=J, T=T)
    # On mesure la magnétisation
    mT_vals.append(aimantation(spins))


# === Tracé : Magnétisation vs Température ===
plt.figure(figsize=(7, 5))
plt.plot(T_vals, mT_vals, "-o", ms=6, color="b")
plt.xlabel("Température T")
plt.ylabel("Aimantation par spin")
plt.title(f"Coubre de l'aimantation en fonction de la température (Modèle d’Ising 2D, L={L})")
plt.grid(True)
plt.show()


# === Boucle d’hystérésis (variation de H à T fixe) ===
T = 1.0  # Température fixée pour la boucle
for H in H_vals:
    # On laisse évoluer le système vers l’équilibre
    for _ in range(n_mcs):
        spins = metropolis(spins, H, J, T)
    # On enregistre la magnétisation
    mH_vals.append(aimantation(spins))


# === Tracé : Boucle d’hystérésis ===
plt.figure(figsize=(7, 5))
plt.plot(H_vals, mH_vals, "-", marker='o', ms=6, color="r")
plt.xlabel("Champ magnétique H")
plt.ylabel("Aimantation par spin")
plt.title(f"Cycle d’hystérésis (Modèle d’Ising 2D, L={L}, T={T})")
plt.grid(True)
plt.show()
