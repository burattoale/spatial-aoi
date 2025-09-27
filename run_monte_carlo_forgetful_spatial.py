import json
import numpy as np
import math
import pickle
from copy import deepcopy

from utils import SimulationParameters, run_monte_carlo_simulation_spatial

def run_simulation_from_json(json_path):
    """
    Runs a Monte Carlo simulation for a forgetful receiver with spatial awareness
    based on parameters from a JSON file.
    """
    with open(json_path, 'r') as f:
        params_json = json.load(f)

    initial_params = SimulationParameters(
        q=params_json['q'],
        eta=params_json['eta'],
        zeta=params_json['zeta'],
        epsilon=params_json['epsilon'],
        rho=params_json['rho'],
        alpha=params_json['alpha'],
        beta=params_json['beta'],
        R_unit=params_json['R_unit'],
        X_symbols=params_json['X_symbols'],
        noise_distribution=params_json['noise_distribution']
    )

    k_min, k_max = params_json['K_range']
    num_simulations = params_json['num_simulations']
    num_steps = params_json['num_steps']
    
    results = {}

    for k in range(k_min, k_max):
        params = deepcopy(initial_params)
        params.K = k
        params.m = math.floor(params.rho * np.pi * (params.K * params.R_unit)**2)
        
        print(f"Running simulation for K={k}, m={params.m}")

        entropies = []
        for i in range(5):
            print(f"  - Run {i+1}/5")
            avg_entropy, _ = run_monte_carlo_simulation_spatial(
                params,
                num_steps,
                num_simulations,
                seed=i,
            )
            entropies.append(avg_entropy)
        
        avg_entropy_final = np.mean(entropies)
        results[k] = avg_entropy_final
        print(f"K={k}, Average Entropy={avg_entropy_final} (averaged over 5 runs)")

    output_filename = "results/forgetful_spatial_binary_spawc.pkl"
    with open(output_filename, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Simulation finished. Results saved to {output_filename}")

if __name__ == "__main__":
    run_simulation_from_json('experiments/binary_spawc.json')
