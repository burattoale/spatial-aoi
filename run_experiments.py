from utils import SimulationParameters

from hmm_joint_prob import run_hmm_simulation
from utils import run_monte_carlo_simulation

from joblib import Parallel, delayed
import numpy as np
import math
from copy import deepcopy
import pickle
from tqdm import tqdm

k_min = 1
k_max = 60
num_simulations = 40

sim_length = 10000

hmm_params = SimulationParameters(
        q=0.005,
        eta=1,
        zeta=5e-4, # Example: list of base probabilities. NodeDistribution will use this.
        epsilon=0.1,
        rho=0.05,
        m_override=None,
        K=2, # Initial K for the first plot
        alpha=0.02,
        beta=0, # Default beta, will be overridden in the loop
        R_unit=5,
        Y_symbols=[0,1,2]
    )

results = {}
results['hmm'] = {}
results['forgetful'] = {}
for eta in tqdm(range(1, 50,1)):
    results['hmm'][f"eta:{eta}"] = {}
    results['forgetful'][f"eta:{eta}"] = {}
    hmm_params.eta = eta
    for k in range(k_min, k_max):
        hmm_params.K = k
        hmm_params.m = math.floor(hmm_params.rho * np.pi * (hmm_params.R_unit*k)**2)
        hmm_res = Parallel(n_jobs=10,)(
            delayed(run_hmm_simulation)(hmm_params, sim_length, seed=i) for i in range(num_simulations)
        )
        averages_hmm = [r[1] for r in hmm_res]
        results['hmm'][f"eta:{eta}"][k] = np.mean(averages_hmm)
        forgetful_params = deepcopy(hmm_params)
        forgetful_params.Y_symbols = [0,1]
        forgetful_res = Parallel(n_jobs=10,)(
            delayed(run_monte_carlo_simulation)(hmm_params, sim_length, 100, seed=i) for i in range(num_simulations)
        )
        averages_forgetful = [r[1] for r in forgetful_res]
        results['forgetful'][f"eta:{eta}"][k] = np.mean(averages_forgetful)
        # save
        with open("results/binary_simulations_eta_R_unit_5.pkl", 'wb') as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)