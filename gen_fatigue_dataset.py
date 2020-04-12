"""Module to randomly generate some fatigue test results."""
import numpy as np
import pandas as pd

from functools import partial
from numpy.random import multivariate_normal

from support_functions import initializer, kwargs_run

def default_init_fcn(s_max, s_min, A_init=7, b_init=12, m_init=0.5):
    return (np.exp(A_init)/(s_max - s_min)*(1 - s_min/s_max)**(m_init-1))**b_init

def default_cp_fcn(s_max, s_min, A_cp=8, b_cp=5, m_cp=0.5):
    return (np.exp(A_cp)/(s_max - s_min)*(1 - s_min/s_max)**(m_cp-1))**b_cp

def generate_pars(fixed_pars, stochastic_pars):
    if stochastic_pars is not {}:
        samples = multivariate_normal(stochastic_pars["mean"], stochastic_pars["cov"])
        stoch = {i: j for i, j in zip(stochastic_pars["names"], samples)}
    return {**fixed_pars, **stoch}

class FatigueTestGenerator:
    @initializer
    def __init__(self, spec_inputs,
        init_fcns=[default_init_fcn], cp_fcn=default_cp_fcn,
        fixed_pars={}, stochastic_pars={}):
        """Initialise generator."""
        self.specimens = {}
        self.current = 0
        self.len = len(self.spec_inputs.index)

    def __getitem__(self, idx):
        """Evaluate specimen idx."""
        if idx not in self.specimens.keys():
            pars = generate_pars(self.fixed_pars, self.stochastic_pars)
            self.specimens[idx] = Specimen(self.spec_inputs.loc[idx],
                self.init_fcns, self.cp_fcn, pars)
        return self.specimens[idx]()
    
    def __next__(self):
        value = self.current
        if value >= len(self):
            raise StopIteration
        self.current += 1
        return self[value]
    
    def __iter__(self):
        self.current = 0
        return self
    
    def __len__(self):
        return self.len

class Specimen:
    @initializer
    def __init__(self, spec_data, init_fcns, cp_fcn, pars):
        pass

    def __call__(self):
        self.init_life = min([kwargs_run(init_fcn,[self.spec_data, self.pars]) for init_fcn in self.init_fcns])
        self.cp_life = kwargs_run(self.cp_fcn, [self.spec_data, self.pars])
        self.total_life = self.init_life + self.cp_life
        return self.total_life

if __name__ == "__main__":
    spec_inputs = pd.DataFrame(columns=["s_max", "s_min", "mat_source", "Kt", "max_cycles", "eval_strain", "eval_pd", "eval_init_type"])
    s_max = [300]*15 + [400]*15 + [500]*15 + [600]*15 + [700]*15 + [800]*15 + [900]*15 + [1000]*15
    stochastic_pars = {"names": ["A_init", "b_init", "A_cp"], "mean": [7, 12, 8.2], "cov": np.array([[0.01, 0, 0], [0, 9, 0], [0, 0, 0.003]])}
    for i in s_max:
        spec_inputs = spec_inputs.append({"s_max": i,"s_min": 0,"max_cycles": 100000},ignore_index=True)
    test_gen = FatigueTestGenerator(spec_inputs, [default_init_fcn], stochastic_pars=stochastic_pars)#, partial(default_init_fcn, A=1e21, b=-6)])
    lives = [i for i in test_gen]
    print(lives)

    import matplotlib.pyplot as plt
    plt.plot([i for i in test_gen], s_max, ".")
    plt.xscale("log")
    plt.show()