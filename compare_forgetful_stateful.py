import math
import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from tqdm import tqdm

from hmm_sim import hmm_entropy
from main_simulation import run_monte_carlo_simulation
from utils import overall_entropy, SimulationParameters
from environment import DTMC, NodeDistribution

def do_simulation_pair(params:SimulationParameters, sim_length:int):
    params.beta = 0
    params.Y_symbols = [0,1,2,3]
    time_evolution, mean_e_hmm = hmm_entropy(params, sim_length)

    params.Y_symbols = [0,1]
    mean_e_forgetful, time_evolution_forgetful = run_monte_carlo_simulation(
        params=params,
        num_time_steps=sim_length,
        num_burn_in_steps=100,
    )
    # plt.figure()
    # plt.plot(time_evolution_forgetful, label="forgetful receiver")
    # plt.plot(time_evolution, label="HMM")
    # plt.legend()
    # plt.show()
    del time_evolution, time_evolution_forgetful


    return mean_e_forgetful, mean_e_hmm

if __name__ == "__main__":
    sim_length = 10000
    n_topologies = 20
    k_max = 60

    q = 0.005
    eta = 1
    zeta = 0.0001
    epsilon = 0.1
    rho = 5e-2
    R = 10
    K = 5
    m = math.floor(rho * np.pi * (K*R)**2)
    alpha = 0.02
    x_symbols = [0, 1]
    y_symbols = [0, 1, 2, 3]
    y_translation = {
        0:"0",
        1:"1",
        2:"C",
        3:"I"
    }

    params = SimulationParameters(q, eta, zeta, epsilon, rho=rho, m=m, K=K, alpha=alpha, R_unit=R, X_symbols=x_symbols, Y_symbols=y_symbols)

    entropies_k_forgetful = []
    entropies_k_hmm = []
    for k in tqdm(range(1,k_max), desc="Working over Ks"):
        params.K = k
        params.m = math.floor(rho * np.pi * (params.R_unit*k)**2)
        results = Parallel(n_jobs=-1, backend="loky", verbose=1)(
            delayed(do_simulation_pair)(params, sim_length) for _ in range(n_topologies)
        )
        # average the results from the run
        averages = np.mean(results, axis=0)
        entropies_k_forgetful.append(averages[0])
        entropies_k_hmm.append(averages[1])

        # for the forgetful receiver create DTMC and tx probability vector
        # mm = DTMC(params.q, params.eta)
        # topology = NodeDistribution(params.rho, params.R_unit, params.K, params.alpha, params.beta)
        # mean_e_forgetful = overall_entropy(A=mm.A, pi=mm.pi, K=params.K, 
        #                                   R_unit=params.R_unit, alpha=params.alpha, m=topology.m, 
        #                                   zeta=topology.tx_probabilities, epsilon=params.epsilon, 
        #                                   states=params.X_symbols, max_delta_considered=1000)

        # entropies_k_forgetful.append(mean_e_forgetful)
        # entropies_k_hmm.append(np.mean(results))

    plt.figure()
    plt.plot(np.arange(1,k_max), entropies_k_forgetful, marker='.', label="Forgetful receiver")
    plt.plot(np.arange(1,k_max), entropies_k_hmm, marker='.', label="HMM receiver")
    plt.xlabel("K")
    plt.ylabel("average estimation entropy")
    plt.legend()
    plt.grid(True)
    plt.show()