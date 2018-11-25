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
        observable_storage = []
        potential_storage = []
        collection = []
        pool = mp.Pool(self.n_proc)
        for _ in range(self.n_sample):
            dmu, Vq = next(param_gen)
            potential_storage.append(np.array([dmu, *Vq]))
            collection.append(pool.apply_async(self.solver, args=(nk, nbasis, dmu, Vq)))
        pool.close()
        pool.join()
        
        for item in collection:
            T, density = item.get()
            observable_storage.append(np.array([T, *density]))
        observable = np.array(observable_storage)
        potential = np.array(potential_storage)

        with open(self.out, 'wb') as f:
            pickle.dump(observable, f)
        with open(self.aux, 'wb') as f1:
            pickle.dump(potential, f1)
            



            
            