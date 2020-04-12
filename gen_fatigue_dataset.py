"""Module to randomly generate some fatigue test results."""
import pandas as pd

from functools import partial

from support_functions import initializer, kwargs_run

def default_init_fcn(s_max, s_min, A=1e21, b=-6, m=0.5):
    return A*((s_max - s_min)*(1 - s_min/s_max)**(m-1))**b

def default_cp_fcn(s_max, s_min, A=0, b=-6, m=0.5):
    return A*((s_max - s_min)*(1 - s_min/s_max)**(m-1))**b

def generate_pars(fixed_pars, stochastic_pars):
    pars = {}
    return pars

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
    for i in [500, 600, 700, 800, 900, 1000]:
        spec_inputs = spec_inputs.append({"s_max": i,"s_min": 0,"max_cycles": 100000},ignore_index=True)
    test_gen = FatigueTestGenerator(spec_inputs, [default_init_fcn, partial(default_init_fcn, A=2e32, b=-10)])
    print([i for i in test_gen])