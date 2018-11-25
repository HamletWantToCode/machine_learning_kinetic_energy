# 1d lattice parallel over kpoints

import numpy as np 
import pickle
import multiprocessing as mp 

class Compute(object):
    def __init__(self, n_proc, n_sample, datafile, auxfile):
        self.n_sample = n_sample
        self.n_proc = n_proc
        self.out = datafile
        self.aux = auxfile

    def add_solver(self, solver):
        self.solver = solver

    def run(self, nk, nbasis, param_gen):
        kpoints = np.linspace(0, np.pi, nk)
        observable_storage = []
        potential_storage = []
        for _ in range(self.n_sample):
            mu, Vq = next(param_gen)
            collection = []
            T = 0
            density = np.zeros(nbasis, dtype=np.complex64)
            pool = mp.Pool(self.n_proc)
            for k in kpoints:
                collection.append(pool.apply_async(self.solver, args=(k, nbasis, mu, Vq)))
            pool.close()
            pool.join()

            for item in collection:
                T_k, density_k = item.get()
                T += T_k
                density += density_k
            T/=nk
            density/=nk
            
            observable_storage.append(np.array([T, *density]))
            potential_storage.append(np.array([mu, *Vq]))
        
        observable = np.array(observable_storage)
        potential = np.array(potential_storage)
        with open(self.out, 'wb') as f:
            pickle.dump(observable, f)
        with open(self.aux, 'wb') as f1:
            pickle.dump(potential, f1)
            



            
            