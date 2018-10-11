# kronig penny model
import numpy as np 

def KP_potential(nG, r0, V0):
    Vx = np.zeros(nG)
    for i in range(nG):
        if i > nG*r0:
            Vx[i] = -V0
    Vq = np.fft.rfft(Vx)/nG
    return Vq

if __name__ == '__main__':
    import matplotlib.pyplot as plt 
    nG = 101
    r0 = 0.7
    V0 = 50
    X = np.linspace(0, 1, nG)
    Vq = KP_potential(nG, r0, V0)
    Vx = np.fft.irfft(Vq, n=nG)*nG
    # Kpoints_positive = np.fft.fftfreq(nG)[:nG//2+1]
    # plt.stem(Kpoints_positive, Vq)
    plt.plot(X, Vx)
    plt.show()
