import numpy as np
import matplotlib.pyplot as plt

# Kích thước hệ
L = 20
N = L * L

# Tham số mô phỏng
J = 1.0
H = 0.0
T = 2.0
beta = 1.0 / T
n_steps = 5000  # số MCSS

# Hàm khởi tạo spin ngẫu nhiên
def init_spins(L):
    return np.random.choice([-1, 1], size=(L, L))

# Tính thay đổi năng lượng khi lật spin
def delta_E(spins, i, j, J, H):
    L = spins.shape[0]
    s = spins[i, j]
    # láng giềng gần nhất (chu kỳ)
    nn = spins[(i+1)%L,j] + spins[(i-1)%L,j] + spins[i,(j+1)%L] + spins[i,(j-1)%L]
    return 2 * s * (J * nn + H)

# Một MCSS
def metropolis_step(spins, beta, J, H):
    L = spins.shape[0]
    for _ in range(L*L):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        dE = delta_E(spins, i, j, J, H)
        if dE <= 0 or np.random.rand() < np.exp(-beta * dE):
            spins[i, j] *= -1
    return spins

# Tính từ hóa trung bình
def magnetization(spins):
    return np.sum(spins) / spins.size

# Mô phỏng
spins = init_spins(L)
m_vals = []

for step in range(n_steps):
    spins = metropolis_step(spins, beta, J, H)
    m_vals.append(magnetization(spins))

# Vẽ đồ thị từ hóa theo "Monte Carlo time"
plt.figure(figsize=(8,5))
plt.plot(range(n_steps), m_vals, color="b")
plt.xlabel("Monte Carlo time (MCSS)")
plt.ylabel("Magnetization per spin")
plt.title(f"Ising 2D: Evolution of magnetization (L={L}, T={T})")
plt.grid(True)
plt.show()
