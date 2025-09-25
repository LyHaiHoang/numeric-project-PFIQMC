import numpy as np
import matplotlib.pyplot as plt

# ---------- Parameters ----------
N = 1000            # số hạt (ví dụ 100, bạn tăng dần sau)
rho = 0.8          # mật độ
L = N / rho        # chiều dài hộp
dt = 0.001         # bước thời gian
nsteps = 500      # số bước mô phỏng
epsilon = 1.0
sigma = 1.0
r_cut = 2.5 * sigma
m = 1.0

# ---------- Functions ----------
def minimum_image(dx, L):
    return dx - L*np.round(dx/L)

def compute_forces(x, L, epsilon, sigma, r_cut):
    N = len(x)
    forces = np.zeros(N)
    potential = 0.0
    rcut2 = r_cut**2
    for i in range(N-1):
        for j in range(i+1, N):
            dx = x[i]-x[j]
            dx = minimum_image(dx, L)
            r2 = dx*dx
            if r2 < rcut2:
                inv_r2 = 1.0/r2
                inv_r6 = (sigma**6)*(inv_r2**3)
                inv_r12 = inv_r6*inv_r6
                vij = 4*epsilon*(inv_r12 - inv_r6)
                fij = 24*epsilon*(2*inv_r12 - inv_r6)*inv_r2
                f = fij*dx
                forces[i] += f
                forces[j] -= f
                potential += vij
    return forces, potential

def velocity_verlet(x, v, m, dt, L, epsilon, sigma, r_cut):
    f, pot = compute_forces(x, L, epsilon, sigma, r_cut)
    v_half = v + 0.5*f/m*dt
    x = x + v_half*dt
    x = np.mod(x, L)  # điều kiện biên tuần hoàn
    f_new, pot_new = compute_forces(x, L, epsilon, sigma, r_cut)
    v = v_half + 0.5*f_new/m*dt
    return x, v, pot_new

# ---------- Initialization ----------
np.random.seed(0)
x = np.linspace(0, L, N, endpoint=False)
v = np.random.normal(0, 0.1, N)
v -= v.mean()

# ---------- Simulation ----------
E_kin = []
E_pot = []
for step in range(nsteps):
    x, v, pot = velocity_verlet(x, v, m, dt, L, epsilon, sigma, r_cut)
    kin = 0.5*m*np.sum(v*v)
    E_kin.append(kin)
    E_pot.append(pot)
    if step % 500 == 0:
        print(f"Step {step}  E_tot={kin+pot:.3f}")

# ---------- Plot energies ----------
plt.plot(E_kin, label='Kinetic')
plt.plot(E_pot, label='Potential')
plt.plot(np.array(E_kin)+np.array(E_pot), label='Total')
plt.xlabel('Step')
plt.ylabel('Energy')
plt.legend()
plt.show()
