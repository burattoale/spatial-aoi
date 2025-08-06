import argparse
import os
from tqdm import tqdm
import numpy as np
import math
import pickle
from copy import deepcopy
from joblib import Parallel, delayed
import json

from utils import SimulationParameters
from hmm_joint_prob import run_hmm_simulation
from utils import run_monte_carlo_simulation
from joblib import Parallel, delayed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the main script with specified arguments.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    
    args = parser.parse_args()

    configs = json.load(open(args.config, 'r'))
    
    # Ensure output directory exists
    os.makedirs("results", exist_ok=True)

    hmm_params = SimulationParameters(
        q=configs['q'],
        eta=configs['eta'],
        zeta=configs['zeta'],
        epsilon=configs['epsilon'],
        rho=configs['rho'],
        alpha=configs['alpha'],
        R_unit=configs['R_unit'],
        noise_distribution=configs['noise_distribution'],
        X_symbols=configs['X_symbols']
        )
    
    eta_min, eta_max = configs['eta_range']
    k_min, k_max = configs['K_range']
    sim_length = configs['num_steps']
    num_simulations = configs['num_simulations']

    NON_BINARY = True if len(hmm_params.X_symbols) > 2 else False
    LOC_AWARE = configs["loc_aware"]

    PARALLEL_JOBS = 10

    results = {}
    results['hmm'] = {}
    results['forgetful'] = {}
    results['hmm_err'] = {}
    for eta in tqdm(range(eta_min, eta_max)):#range(1, 50,1)):
        results['hmm'][f"eta:{eta}"] = {}
        results['forgetful'][f"eta:{eta}"] = {}
        results['hmm_err'][f"eta:{eta}"] = {}
        hmm_params.eta = eta
        for k in range(k_min, k_max):
            hmm_params.K = k
            hmm_params.m = math.floor(hmm_params.rho * np.pi * (hmm_params.R_unit*k)**2)
            if LOC_AWARE:
                hmm_params.Y_symbols = [i for i in range(hmm_params.K * len(hmm_params.X_symbols) +1)]
            else:
                hmm_params.Y_symbols = deepcopy(hmm_params.X_symbols)
                hmm_params.Y_symbols.append(len(hmm_params.X_symbols)) # add no receive symbol
            hmm_res = Parallel(n_jobs=PARALLEL_JOBS,)(
                delayed(run_hmm_simulation)(hmm_params, sim_length, seed=i, loc_aware=LOC_AWARE, non_binary=NON_BINARY) for i in range(num_simulations)
            )
            averages_hmm = [r[1] for r in hmm_res]
            averages_hmm_est_error = [r[2] for r in hmm_res]
            results['hmm'][f"eta:{eta}"][k] = np.mean(averages_hmm)
            results['hmm_err'][f"eta:{eta}"][k] = np.mean(averages_hmm_est_error)
            # forgetful
            # do not do the locaion aware stuff
            if not LOC_AWARE and not NON_BINARY:
                forgetful_params = deepcopy(hmm_params)
                forgetful_params.Y_symbols = deepcopy(forgetful_params.X_symbols)
                forgetful_res = Parallel(n_jobs=PARALLEL_JOBS,)(
                    delayed(run_monte_carlo_simulation)(forgetful_params, sim_length, 100, seed=i) for i in range(num_simulations)
                )
                averages_forgetful = [r[1] for r in forgetful_res]
                results['forgetful'][f"eta:{eta}"][k] = np.mean(averages_forgetful)
            # save
            with open(configs["output_file"], 'wb') as f:
                pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)