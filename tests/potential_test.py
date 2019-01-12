# potential test

import numpy as np 
from MLEK.main.utils import potential_gen, irfft
import matplotlib.pyplot as plt 

np.random.seed(8)

V0 = 10
X = np.linspace(0, 1, 100)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
# simple
theta = np.random.uniform(0, 2*np.pi)
Vq_simple = np.array([0, -V0*(np.cos(theta) + 1j*np.sin(theta))])
Vx_simple = irfft(Vq_simple, 100)
ax1.plot(X, Vx_simple, 'b', label='simplest')

# complicated
Vq_complex = np.zeros(5, dtype=np.complex64)
for i in range(1, 5):
    theta = np.random.uniform(0, 2*np.pi) 
    r0 = np.random.rand()
    Vq_complex[i] = -V0*r0*(np.cos(theta) + 1j*np.sin(theta)) 
Vx_complex = irfft(Vq_complex, 100)
ax1.plot(X, Vx_complex, 'g', label='complicated')

# general view
nbasis = 5
max_q = 5
low_V0 = 1
high_V0 = 10
dmu = 5
gen = potential_gen(nbasis, max_q, low_V0, high_V0, dmu, 8)
for i in range(20):
    _, Vq, _ = next(gen)
    Vx = irfft(Vq, 100)
    ax2.plot(X, Vx)

ax1.set_xlabel('x')
ax1.set_ylabel('V(x)')
ax1.legend()
ax2.set_xlabel('x')
plt.savefig('data_file/potential_demo.png')