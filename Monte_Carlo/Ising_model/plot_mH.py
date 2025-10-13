import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

# --- Chọn nhiều file dữ liệu ---
root = Tk()
root.withdraw()
filenames = filedialog.askopenfilenames(
    title="Chọn file dữ liệu hystérésis",
    filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
)

# --- Vẽ đồ thị ---
plt.figure(figsize=(7, 5))

for filename in filenames:
    data = np.loadtxt(filename)
    H_vals = data[:, 0]   # cột 1 = H
    mH_vals = data[:, 1]  # cột 2 = m
    plt.plot(H_vals, mH_vals, "-s", ms=4, label=filename.split("/")[-1])

plt.xlabel("Champ magnétique H")
plt.ylabel("Aimantation par spin")
plt.title("Cycles d’hystérésis (Modèle d’Ising 2D)")
plt.legend()
plt.grid(True)
plt.show()
