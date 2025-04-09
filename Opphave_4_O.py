import numpy as np
import matplotlib.pyplot as plt

# PARAMETERE 
L = 1.0          
T = 0.1          
n = 20           
h = 0.05
k = 0.002
m = int(T / k)   

# LAG GITTER
x = np.linspace(0, L, n+1)     
t = np.linspace(0, T, m+1)     
alpha = k / h**2     


# INITIALISER LØSNINGSMATRISE 
u = np.zeros((m+1, n+1))       
u[0, :] = np.sin(x)    

# LØS MED EKSPLOSITT SKJEMA 
for j in range(m):                    
    for i in range(1, n):             
        u[j+1, i] = u[j, i] + alpha * (u[j, i+1] - 2*u[j, i] + u[j, i-1])

# PLOTT RESULTATET
plt.figure(figsize=(8, 5))
for j in range(0, m+1, m//5):         
    plt.plot(x, u[j], label=f"t = {t[j]:.3f}")
plt.title("Varmelikningen med eksplisitt metode")
plt.xlabel("x")
plt.ylabel("Temperatur u(x,t)")
plt.legend()
plt.grid(True)
plt.show()
