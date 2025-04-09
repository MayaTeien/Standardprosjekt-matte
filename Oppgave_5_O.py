import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

# Parametre
L = 1.0  # Lengde på stangen
T = 0.5  # Sluttid
n = 20   
m = 100  

h = 0.05
k = 0.0001
r = k / h**2


x = np.linspace(0, L, n + 2)
t = np.linspace(0, T, m + 1)


u = np.zeros((n + 2, m + 1))
u[:, 0] = np.sin(x)


main_diag = (1 + 2*r) * np.ones(n)
off_diag = -r * np.ones(n - 1)
A = np.diag(main_diag) + np.diag(off_diag, -1) + np.diag(off_diag, 1)


for j in range(0, m):
    b = u[1:-1, j]  # høyreside
    u[1:-1, j+1] = solve(A, b)


plt.plot(x, u[:, 0], label='t=0')
plt.plot(x, u[:, m//2], label=f't={T/2:.2f}')
plt.plot(x, u[:, -1], label=f't={T:.2f}')
plt.legend()
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.title('Implisitt Euler-metode for varmelikningen\nmed u(x,0) = sin(x)')
plt.grid(True)
plt.show()