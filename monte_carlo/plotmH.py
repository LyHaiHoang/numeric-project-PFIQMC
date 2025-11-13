import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import re

# --- Chọn nhiều file dữ liệu ---
root = Tk()
root.withdraw()
filenames = filedialog.askopenfilenames(
    title="Chọn file dữ liệu hystérésis",
    filetypes=[("Text files", "*.dat"), ("All files", "*.*")]
)

# --- Vẽ đồ thị ---
plt.figure(figsize=(7, 5))

for filename in filenames:
    data = np.loadtxt(filename)
    H_vals = data[:, 0]   # cột 1 = H
    mH_vals = data[:, 1]  # cột 2 = m

    # --- Trích xuất giá trị T từ tên file ---
    match = re.search(r'T([\d\.]+)', filename)
    if match:
        T_val = match.group(1).rstrip('.')  # loại dấu chấm cuối nếu có
        label_text = f"T = {T_val}"
    else:
        label_text = "T không xác định"

    plt.plot(H_vals, mH_vals, "-o", ms=4, label=label_text)

plt.xlabel("Champ magnétique H")
plt.ylabel("Aimantation par spin")
plt.title("Cycles d’hystérésis (Modèle d’Ising 2D)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
