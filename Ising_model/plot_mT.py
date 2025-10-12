import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

# --- Chọn nhiều file dữ liệu ---
root = Tk()
root.withdraw()
filenames = filedialog.askopenfilenames(
    title="Chọn file dữ liệu T–m",
    filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
)

# --- Vẽ đồ thị ---
plt.figure(figsize=(7, 5))

for filename in filenames:
    data = np.loadtxt(filename)
    T_vals = data[:, 0]   # cột 1 = T
    mT_vals = data[:, 1]  # cột 2 = m
    plt.plot(T_vals, mT_vals, "-s",color="blue", ms=4, label="L=100")#label=filename.split("/")[-1])

plt.xlabel("Température T")
plt.ylabel("Aimantation par spin")
plt.title("Courbes de l'aimantation en fonction de la température (Modèle d’Ising 2D)")
plt.legend()
plt.grid(True)
plt.show()
