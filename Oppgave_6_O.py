import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
from matplotlib.animation import FuncAnimation

# Parametre
L = 1.0
T = 0.5
n = 20
m = 200
h = 0.05
k = 0.001
alpha = k / h**2

x = np.linspace(0, L, n + 2)
t = np.linspace(0, T, m + 1)
u0 = np.sin(x) 


# Eksplisitt Euler
u_exp = np.zeros((n + 2, m + 1))
u_exp[:, 0] = u0.copy()
for j in range(m):
    for i in range(1, n + 1):
        u_exp[i, j+1] = u_exp[i, j] + alpha * (u_exp[i+1, j] - 2*u_exp[i, j] + u_exp[i-1, j])


# Implisitt Euler
u_impl = np.zeros((n + 2, m + 1))
u_impl[:, 0] = u0.copy()
main_diag = (1 + 2*alpha) * np.ones(n)
off_diag = -alpha * np.ones(n - 1)
A_impl = np.diag(main_diag) + np.diag(off_diag, -1) + np.diag(off_diag, 1)
for j in range(m):
    b = u_impl[1:-1, j]
    u_impl[1:-1, j+1] = solve(A_impl, b)


# Crank-Nicolson
u_cn = np.zeros((n + 2, m + 1))
u_cn[:, 0] = u0.copy()
main_A = (1 + alpha) * np.ones(n)
off_A = -alpha/2 * np.ones(n - 1)
A_cn = np.diag(main_A) + np.diag(off_A, -1) + np.diag(off_A, 1)
main_B = (1 - alpha) * np.ones(n)
off_B = alpha/2 * np.ones(n - 1)
B_cn = np.diag(main_B) + np.diag(off_B, -1) + np.diag(off_B, 1)
for j in range(m):
    b = B_cn @ u_cn[1:-1, j]
    u_cn[1:-1, j+1] = solve(A_cn, b)

# Analytisk løsning
u_exact = np.outer(np.sin(x), np.exp(-t))

# Plot sammenligning ved sluttidspunktet
plt.figure(figsize=(10, 6))
plt.plot(x, u_exact[:, -1], label="Analytisk", linestyle='--', color='black')
plt.plot(x, u_exp[:, -1], label="Eksplisitt Euler")
plt.plot(x, u_impl[:, -1], label="Implisitt Euler")
plt.plot(x, u_cn[:, -1], label="Crank–Nicolson")
plt.xlabel("x")
plt.ylabel("u(x,T)")
plt.title("Sammenligning av numeriske metoder og analytisk løsning")
plt.legend()
plt.grid()
plt.show()


# Animasjon over tid
fig, ax = plt.subplots()
line1, = ax.plot(x, u_exact[:, 0], 'k--', label='Analytisk')
line2, = ax.plot(x, u_cn[:, 0], 'r', label='Crank–Nicolson')
line3, = ax.plot(x, u_impl[:, 0], 'g', label='Implisitt')
line4, = ax.plot(x, u_exp[:, 0], 'b', label='Eksplisitt')
ax.set_ylim(-0.1, 1.1)
ax.set_title("Tid = 0.00")
ax.legend()
ax.set_xlabel("x")
ax.set_ylabel("u(x,t)")

def animate(j):
    line1.set_ydata(u_exact[:, j])
    line2.set_ydata(u_cn[:, j])
    line3.set_ydata(u_impl[:, j])
    line4.set_ydata(u_exp[:, j])
    ax.set_title(f"Tid = {t[j]:.2f}")
    return line1, line2, line3, line4

ani = FuncAnimation(fig, animate, frames=range(0, m+1, 4), interval=100)
plt.show()