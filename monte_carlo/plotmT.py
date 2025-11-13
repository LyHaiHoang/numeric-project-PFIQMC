import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import re

root = Tk()
root.withdraw()
filenames = filedialog.askopenfilenames(
    title="Sélectionner les fichiers de données T–M",
    filetypes=[("Fichiers texte", "*.dat"), ("Tous les fichiers", "*.*")]
)

plt.figure(figsize=(7, 5))

for filename in filenames:
    data = np.loadtxt(filename)
    T_vals, mT_vals = data[:, 0], data[:, 1]
    plt.plot(T_vals, mT_vals, "-s", ms=3, label=None)

Tc = 2 / np.log(1 + np.sqrt(2))
plt.scatter(Tc, 0, marker='s', facecolors='none', edgecolors='red', s=50, linewidths=1.5)

plt.annotate(
    r"$T_c \approx 2.27$",
    xy=(Tc, 0), xycoords='data',
    xytext=(Tc - 0.4, 0.25),
    textcoords='data',
    arrowprops=dict(facecolor='red', edgecolor='red',
                    arrowstyle='->', lw=1.5),
    fontsize=10, color='red'
)

plt.xlabel("Température")
plt.ylabel("Aimantation moyenne par spin")
plt.title("Courbe d’aimantation en fonction de la température du modèle d’Ising 2D")
plt.grid(True, linestyle='--', alpha=1)
plt.tight_layout()
plt.show()
