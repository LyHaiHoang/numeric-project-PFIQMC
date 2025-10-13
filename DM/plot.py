import numpy as np
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file (bỏ dòng tiêu đề có dấu #)
filename = "energies.dat"
data = np.loadtxt(filename, comments='#') 


# Nếu bạn đã có file energies.dat, hãy thay bằng:
# data = np.loadtxt(filename, comments='#')

# Tách cột
step = data[:, 0]
kinetic = data[:, 1]
potential = data[:, 2]
total = data[:, 3]

# Vẽ
plt.figure(figsize=(8,5))
plt.plot(step, kinetic, label='Kinetic Energy', linewidth=2)
plt.plot(step, potential, label='Potential Energy', linewidth=2)
plt.plot(step, total, label='Total Energy', linewidth=2, linestyle='--', color='black')

plt.xlabel('Step')
plt.ylabel('Energy')
plt.title('Energy vs Step (1D Molecular Dynamics)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
