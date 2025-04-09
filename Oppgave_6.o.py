import numpy as np
import matplotlib.pyplot as plt

# Parametere
L = 1.0       # Lengde på stang
T = 0.1     
h = 0.1
k = 0.001
n = int(L / h)      # Rompunkter
m = int(T / k) 
alpha = k / h**2

x = np.linspace(0, L, n+1)
t = np.linspace(0, T, m+1)

# Initialbetingelse
def initial_condition(x): return np.sin(x)


def crank_nicolson():
    u = np.zeros((n+1, m+1))
    u[:, 0] = initial_condition(x)

    a = -alpha/2
    b = 1 + alpha
    c = -alpha/2

    A = np.diag([b]*(n-1)) + np.diag([a]*(n-2), 1) + np.diag([c]*(n-2), -1)
    B = np.diag([1 - alpha]*(n-1)) + np.diag([alpha/2]*(n-2), 1) + np.diag([alpha/2]*(n-2), -1)

    for j in range(m):
        rhs = B @ u[1:n, j]
        u[1:n, j+1] = np.linalg.solve(A, rhs)
    
    return u


def eksplisitt_euler():
    u = np.zeros((n+1, m+1))
    u[:, 0] = initial_condition(x)

    for j in range(m):
        for i in range(1, n):
            u[i, j+1] = u[i, j] + alpha * (u[i+1, j] - 2*u[i, j] + u[i-1, j])
    
    return u


def implisitt_euler():
    u = np.zeros((n+1, m+1))
    u[:, 0] = initial_condition(x)

    A = np.diag([1 + 2*alpha]* (n-1)) + np.diag([-alpha]*(n-2), 1) + np.diag([-alpha]*(n-2), -1)

    for j in range(m):
        u[1:n, j+1] = np.linalg.solve(A, u[1:n, j])
    
    return u



def analytical_solution(x, t):
    return np.exp(-t) * np.sin(x)



U_exp = eksplisitt_euler()
U_imp = implisitt_euler()
U_cn  = crank_nicolson()
U_ana = analytical_solution(x, T)

plt.figure(figsize=(12, 6))
plt.plot(x, U_exp[:, -1], '--', label="Eksplisitt Euler")
plt.plot(x, U_imp[:, -1], '-.', label="Implisitt Euler")
plt.plot(x, U_cn[:, -1], '-', label="Crank–Nicolson", linewidth=2)
plt.plot(x, U_ana, 'k:', label="Analytisk", linewidth=2)

plt.title("Sammenlikning av numeriske metoder med analytisk løsning ved t = 0.5")
plt.xlabel("x")
plt.ylabel("u(x, t)")
plt.legend()
plt.grid()
plt.show()

