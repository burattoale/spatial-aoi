import numpy as np
import optuna
from functools import partial
import math
import pickle
import copy

from main_simulation import run_monte_carlo_simulation
from utils import SimulationParameters, overall_entropy
from environment import DTMC, NodeDistribution
from hmm_joint_prob import run_hmm_simulation

def objective(trial:optuna.Trial, params:SimulationParameters):
    local_params = copy.deepcopy(params)
    lower = 0
    upper = 5e-4
    value = trial.suggest_float("zeta_last", lower, upper)
    zeta = [upper]*(local_params.K-1)
    if local_params.K == 14:
        zeta[-1] = 5e-4
    zeta.append(value)
    local_params.zeta = np.array(zeta)

    # using Monte Carlo simulation
    # entropy, _ = run_monte_carlo_simulation(local_params, 10000, 100, seed=0, zeta_bucket=True) # fix the seed to actually optimize the zeta

    # using formulas and averaging 10 different topologies
    #dtmc = DTMC(local_params.q, local_params.eta)
    NUM_RUNS = 15
    entropies = np.empty(NUM_RUNS, dtype=float)
    for j in range(NUM_RUNS):
        #node_dist = NodeDistribution(
        #    rho=local_params.rho,
        #    unit_radius=local_params.R_unit,
        #    K=local_params.K,
        #    zeta=zeta,
        #    alpha=local_params.alpha,
        #    beta=local_params.beta,
        #    seed=j,
        #    zeta_bucket=True,
        #    fixed_nodes_per_region=True
        #)
        # entropy, _ = run_monte_carlo_simulation(local_params, 10000, 100, seed=0, zeta_bucket=True) # fix the seed to actually optimize the zeta
        #entropy = overall_entropy(A=dtmc.A,
        #                          pi=dtmc.pi,
        #                          K=local_params.K,
        #                          R_unit=local_params.R_unit,
        #                          alpha=local_params.alpha,
        #                          m=len(node_dist),
        #                          zeta=node_dist.tx_probabilities,
        #                          epsilon=local_params.epsilon,
        #                          prob_per_bucket=zeta,
        #                          max_delta_considered=10000)
        _, entropy, _, _, _ = run_hmm_simulation(local_params, 10000, fixed_nodes_per_region=True, seed=j)
        entropies[j] = entropy
        trial.report(entropies[j], j)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return np.mean(entropies)

if __name__ == "__main__":
    initial_params = SimulationParameters(
        q=0.005,
        eta=1,
        zeta=5e-4, # Example: list of base probabilities. NodeDistribution will use this.
        epsilon=0.1,
        rho=0.05,
        m_override=None,
        K=2, # Initial K for the first plot
        alpha=0.02,
        beta=0, # Default beta, will be overridden in the loop
        R_unit=10
    )
    initial_params.m = math.floor(initial_params.rho * np.pi * (initial_params.K*initial_params.R_unit)**2)
    wrapped_objective = partial(objective, params=initial_params)

    results = {}

    last_best_params = None
    for k in range(13, 15):
        current_params = copy.deepcopy(initial_params)
        current_params.K = k
        current_params.m = math.floor(current_params.rho * np.pi * (current_params.K*current_params.R_unit)**2)
        wrapped_objective = partial(objective, params=current_params)

        # call optuna
        study = optuna.create_study(direction="minimize", 
                                    #sampler=optuna.samplers.RandomSampler(), 
                                    pruner=optuna.pruners.MedianPruner()
                                    )
        # warm start the optimization
        values_to_try = [0, 5e-4, 5e-5, 5e-6, 5e-7, 5e-8, 5e-9]
        initial_guesses = {f"params_{n}": {f"zeta_last": val} for n, val in enumerate(values_to_try)}
        initial_guesses["last_best"] = last_best_params if last_best_params is not None else {f"zeta_{i}":9e-5 for i in range(k)}
        for params in initial_guesses.values():
            study.enqueue_trial(params)

        # start the optimization
        if k < 9:
            study.optimize(wrapped_objective, n_trials=200, n_jobs=2)
        else:
            study.optimize(wrapped_objective, n_trials=200, n_jobs=2)
        # compute the entropy for the fixed zeta case zeta = 1e-4
        if isinstance(initial_params.zeta, float):
            current_params.zeta = np.array([initial_params.zeta] * k)
        fixed_entropies = []
        for i in range(15):
            _, entropy, _, _, _ = run_hmm_simulation(current_params, 10000, seed=i, fixed_nodes_per_region=True)
            fixed_entropies.append(entropy)
        entropy = np.mean(fixed_entropies)
        # get the best parameters and save them with the best objective
        results[k] = {"zeta": study.best_params,
                      "entropy_opt": study.best_value,
                      "entropy_same_z": entropy
                      }
        last_best_params = study.best_params
        last_best_params[f"zeta_last"] = 0
        
    print(results)