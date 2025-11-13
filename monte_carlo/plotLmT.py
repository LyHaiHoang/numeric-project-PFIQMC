import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import os

# --- Chọn file dữ liệu có sẵn ---
root = Tk()
root.withdraw()
filenames = filedialog.askopenfilenames(
    title="Sélectionner les fichiers de données Lm(T)",
    filetypes=[("Fichiers texte", "*.dat"), ("Tous les fichiers", "*.*")]
)

plt.figure(figsize=(7, 5))

Tc = 2 / np.log(1 + np.sqrt(2))  # Température critique

for filename in filenames:
    data = np.loadtxt(filename)
    T_vals, Lm_vals = data[:, 0], data[:, 1]

    # Lấy tên file làm nhãn
    label_text = os.path.basename(filename).replace('_', ' ').replace('.dat','')

    plt.plot(T_vals, Lm_vals, "-s", lw=1.5, ms=4, label=label_text)

# Vẽ điểm nhiệt độ tới hạn
plt.scatter(Tc, 0, marker='s', facecolors='none', edgecolors='red', s=50, linewidths=1.5)
plt.annotate(
    r"$T_c \approx 2.27$",
    xy=(Tc, 0), xycoords='data',
    xytext=(Tc-0.5, max(Lm_vals)*0.2),
    textcoords='data',
    arrowprops=dict(facecolor='red', edgecolor='red',
                    arrowstyle='->', lw=1.5),
    fontsize=10, color='red'
)

plt.xlabel("Température T")
plt.ylabel("Chaleur latente $L_m(T)$")
plt.title("Variation de la chaleur latente en fonction de la température")
plt.grid(True, linestyle='--', alpha=0.8)
# plt.legend()
plt.tight_layout()
plt.show()
