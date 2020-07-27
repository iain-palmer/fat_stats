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
    """This class is used to randomly generate fatigue test results."""
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
    s_max = np.array([400]*6 + [500]*6 + [600]*6 + [700]*6 + [800]*6 + [900]*6 + [1000]*6 + [1100]*6 + [1200]*6)
    s_max_rneg1 = np.array([i/2 for i in s_max])
    s_min_rneg1 = np.array([-1*i for i in s_max_rneg1])
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
    import seaborn as sns
    fig, ax = plt.subplots()
    r0_results = np.array([i for i in test_gen_r0])
    rneg1_results = np.array([i for i in test_gen_rneg1])
    sns.scatterplot(x=r0_results, y=s_max, label='R=0 data', ax=ax)
    sns.scatterplot(x=rneg1_results, y=s_max_rneg1, label='R=-1 data', ax=ax)
    plt.xscale("log")
    #plt.show()

    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

    # Define kernel parameters. 
    length_scale = 1000
    noise_level = 1e-5

    # Define kernel object. 
    kernel = 0.5*RBF(length_scale=length_scale)+WhiteKernel(noise_level=noise_level)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    X = s_max.reshape(54, 1)
    x_star = np.linspace(start=400, stop=1200, num=80)
    X_star = x_star.reshape(80, 1)

    gp.fit(X, np.log(r0_results))
    print(gp.log_marginal_likelihood())
    y_mean, y_std = gp.predict(X_star, return_std=True)
    y_mean = np.exp(y_mean)
    
    y_true_normal = np.array([default_init_fcn(i, 0, A_init=7.2, b_init=20) + default_cp_fcn(i, 0, A_cp=8.2) for i in x_star])
    y_true_defect = np.array([default_init_fcn_defect(i, 0, A_defect=7.5, b_defect=12) + default_cp_fcn(i, 0, A_cp=8.2) for i in x_star])
    fig, ax = plt.subplots()
    # Plot true
    sns.lineplot(x=x_star, y=y_true_normal, color='red', label='true_normal', ax=ax)
    sns.lineplot(x=x_star, y=y_true_defect, color='purple', label='true_defect', ax=ax)
    # Plot results.
    sns.scatterplot(x=s_max, y=r0_results, label='R=0 data', ax=ax)
    # Plot prediction. 
    sns.lineplot(x=x_star, y=y_mean, color='green', label='pred', ax=ax)
    plt.fill_between(x_star, y_mean*y_std, y_mean/y_std, color='darkorange',
                 alpha=0.2)
    ax.set(title=f'Prediction GaussianProcessRegressor, length_scale = {length_scale}')
    ax.legend(loc='upper right')
    plt.yscale("log")
    plt.show()