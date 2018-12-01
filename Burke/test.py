import numpy as np 
from main import *
import matplotlib.pyplot as plt 

def harmonicPotential(omega):
    def function(x):
        return 0.5*(omega**2)*(x**2)
    return function 

def hydrogenPotential(x):
    return -1.0 / (x + 1e-8)

def squarewllPotential(x):
    return 0

N_points = 100
N_electron = 2

# harmonic oscillator
x_start, x_end = -2.5, 2.5
omega = 5*np.pi  
potential = harmonicPotential(omega)

# hydrogen atom
# x_start, x_end = 0, 10
# potential = hydrogenPotential

# kinetic energy
x_start, x_end = 0, 1
potential = squarewllPotential

H = finiteDifferenceMatrix(N_points, x_start, x_end, potential)
_, eigenfunctions = np.linalg.eigh(H)
compute_density = electronDensity(eigenfunctions, N_electron, N_points, x_start, x_end)
compute_KineticEnergy = kineticEnergy(N_points, x_start, x_end, N_electron, eigenfunctions)

X = np.linspace(x_start, x_end, N_points+2, endpoint=True)

# harmonic oscillator
real_density_1 = np.sqrt(omega/np.pi)*np.exp(-omega*X**2)
real_density_2 = 2*np.sqrt(omega/np.pi)*omega*(X**2)*np.exp(-omega*X**2)
two_electron = real_density_1 + real_density_2

# hydrogen atom

# kinetic energy 
real_density = 2*(np.sin(np.pi*X))**2 + 2*(np.sin(2*np.pi*X))**2
real_KineticEnergy = np.pi**2/2 + 2*np.pi**2

print(real_KineticEnergy)
print(compute_KineticEnergy)

plt.plot(X, real_density, 'r')
plt.plot(X, compute_density, 'b-.')
plt.show()