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
    zeta = []
    lower = 0
    upper = 5e-4
    for i in range(local_params.K): # use m if you want to optimize for the single node and not by region
        value = trial.suggest_float(f"zeta_{i}", lower, upper)
        # fast fail in case everything breaks
        if value > upper:
            raise optuna.TrialPruned()
        zeta.append(value)
        upper = value
    
    zeta = np.array(zeta)
    local_params.zeta = zeta
    # using Monte Carlo simulation
    # entropy, _ = run_monte_carlo_simulation(local_params, 10000, 100, seed=0, zeta_bucket=True) # fix the seed to actually optimize the zeta

    # using formulas and averaging 10 different topologies
    dtmc = DTMC(local_params.q, local_params.eta)
    entropies = np.empty(3, dtype=float)
    for j in range(3):
        node_dist = NodeDistribution(
            rho=local_params.rho,
            unit_radius=local_params.R_unit,
            K=local_params.K,
            zeta=zeta,
            alpha=local_params.alpha,
            beta=local_params.beta,
            seed=j,
            zeta_bucket=True
        )
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
        _, entropy, _, _, _ = run_hmm_simulation(local_params, 10000)
        trial.report(entropy, j)
        if trial.should_prune():
            raise optuna.TrialPruned()
        entropies[j] = entropy
    print(entropies)
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
    for k in range(1, 21):
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
        initial_guesses = {f"params_{n}": {f"zeta_{i}":n * 1e-4 for i in range(k)} for n in range(1,6)}
        initial_guesses["last_best"] = last_best_params if last_best_params is not None else {f"zeta_{i}":9e-5 for i in range(k)}
        for params in initial_guesses.values():
            study.enqueue_trial(params)

        # start the optimization
        if k < 9:
            study.optimize(wrapped_objective, n_trials=1000, n_jobs=2)
        else:
            study.optimize(wrapped_objective, n_trials=1000, n_jobs=2)
        # compute the entropy for the fixed zeta case zeta = 1e-4
        if isinstance(initial_params.zeta, float):
            current_params.zeta = np.array([initial_params.zeta] * k)
        _, entropy, _, _, _ = run_hmm_simulation(current_params, 10000, seed=0)
        # get the best parameters and save them with the best objective
        results[k] = {"zeta": study.best_params,
                      "entropy_opt": study.best_value,
                      "entropy_same_z": entropy
                      }
        last_best_params = study.best_params
        last_best_params[f"zeta_{k}"] = 0
        
        # save partial results
        with open("results/zeta_optim_hmm_12.pickle", "wb") as f:
            pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)