"""Module to randomly generate some fatigue test results."""
import numpy as np
import pandas as pd

from functools import partial
from numpy.random import multivariate_normal
from scipy.stats import norm

from support_functions import initializer, kwargs_run

def default_init_fcn(s_max, s_min, A_init=7.2, b_init=12, m_init=0.5):
    return (np.exp(A_init)/((s_max - s_min)*(1 - s_min/s_max)**(m_init-1)))**b_init

def default_init_fcn_defect(s_max, s_min, z_defect=0.0, A_defect=7.4, b_defect=12, m_defect=0.5, p_thresh=0.5):
    z_thresh = norm.ppf(p_thresh)
    if z_defect <= z_thresh:
        return (np.exp(A_defect)/((s_max - s_min)*(1 - s_min/s_max)**(m_defect-1)))**b_defect
    else:
        return np.inf

def default_cp_fcn(s_max, s_min, A_cp=8, b_cp=5, m_cp_pos=0.5, m_cp_neg=0.2):
    if s_min > 0:
        return (np.exp(A_cp)/((s_max - s_min)*(1 - s_min/s_max)**(m_cp_pos-1)))**b_cp
    else:
        return (np.exp(A_cp)/((s_max - s_min)*(1 - s_min/s_max)**(m_cp_neg-1)))**b_cp

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
    spec_inputs_r0 = pd.DataFrame(columns=["s_max", "s_min", "mat_source", "Kt", "max_cycles", "eval_strain", "eval_pd", "eval_init_type"])
    spec_inputs_rneg1 = pd.DataFrame(columns=["s_max", "s_min", "mat_source", "Kt", "max_cycles", "eval_strain", "eval_pd", "eval_init_type"])
    s_max = [400]*6 + [500]*6 + [600]*6 + [700]*6 + [800]*6 + [900]*6 + [1000]*6 + [1100]*6 + [1200]*6
    s_max_rneg1 = [i/2 for i in s_max]
    s_min_rneg1 = [-1*i for i in s_max_rneg1]
    names = ["A_init", "b_init", "A_cp", "z_defect", "A_defect", "b_defect"]
    mean = [7.2, 20, 8.2, 0, 7.5, 12]
    rho_init = -2
    sigma_A_init = 0.025
    sigma_b_init = 0.25
    sigma_A_cp = 0.055
    rho_defect = -3
    sigma_A_defect = 0.025
    sigma_b_defect = 0.15
    cov = np.array([
        [sigma_A_init**2, rho_init*sigma_A_init*sigma_b_init, 0, 0, 0, 0],
        [rho_init*sigma_A_init*sigma_b_init, sigma_b_init**2, 0, 0, 0, 0],
        [0, 0, sigma_A_cp**2, 0, 0, 0], 
        [0, 0, 0, 1, 0, 0], 
        [0, 0, 0, 0, sigma_A_defect**2, rho_defect*sigma_A_defect*sigma_b_defect],
        [0, 0, 0, 0, rho_defect*sigma_A_defect*sigma_b_defect,  sigma_b_defect**2]
    ])
    stochastic_pars = {"names": names, "mean": mean, "cov": cov}
    for i, j, k in zip(s_max, s_max_rneg1, s_min_rneg1):
        spec_inputs_r0 = spec_inputs_r0.append({"s_max": i,"s_min": 0, "max_cycles": 100000}, ignore_index=True)
        spec_inputs_rneg1 = spec_inputs_rneg1.append({"s_max": j,"s_min": k, "max_cycles": 100000}, ignore_index=True)
    test_gen_r0 = FatigueTestGenerator(spec_inputs_r0, [default_init_fcn, default_init_fcn_defect], stochastic_pars=stochastic_pars)
    test_gen_rneg1 = FatigueTestGenerator(spec_inputs_rneg1, [default_init_fcn, default_init_fcn_defect], stochastic_pars=stochastic_pars)
    np.random.seed(1)
    
    import matplotlib.pyplot as plt
    plt.plot([i for i in test_gen_r0], s_max, "x")
    plt.plot([i for i in test_gen_rneg1], s_max_rneg1, "r.")
    plt.xscale("log")
    plt.show()