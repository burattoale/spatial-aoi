import math
import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from tqdm import tqdm

from hmm_sim import hmm_entropy
from utils import overall_entropy, SimulationParameters
from environment import DTMC, NodeDistribution

def do_simulation_pair(params:SimulationParameters, sim_length:int):
    _, mean_e_hmm = hmm_entropy(params, sim_length)

    del _

    return mean_e_hmm

if __name__ == "__main__":
    sim_length = 1000
    n_topologies = 25

    q = 0.01
    eta = 1
    zeta = 0.002
    epsilon = 0.1
    rho = 5e-2
    R = 10
    K = 5
    m = math.floor(rho * np.pi * R**2*K)
    alpha = 0.02
    x_symbols = [0, 1]
    y_symbols = [0, 1, 2, 3]
    y_translation = {
        0:"0",
        1:"1",
        2:"C",
        3:"I"
    }

    params = SimulationParameters(q, eta, zeta, epsilon, m=m, K=K, alpha=alpha, R_unit=R, X_symbols=x_symbols, Y_symbols=y_symbols)

    entropies_k_forgetful = []
    entropies_k_hmm = []
    for k in tqdm(range(1,12), desc="Working over Ks"):
        params.K = k
        params.m = math.floor(rho * np.pi * params.R_unit**2 * params.K)
        results = Parallel(n_jobs=-1, backend="loky", verbose=1)(
            delayed(do_simulation_pair)(params, sim_length) for _ in range(n_topologies)
        )
        # for the forgetful receiver create DTMC and tx probability vector
        mm = DTMC(params.q, params.eta)
        topology = NodeDistribution(params.rho, params.R_unit, params.K, params.alpha, params.beta)
        mean_e_forgetful = overall_entropy(A=mm.A, pi=mm.pi, K=params.K, 
                                           R_unit=params.R_unit, alpha=params.alpha, m=topology.m, 
                                           zeta=topology.tx_probabilities, epsilon=params.epsilon, 
                                           states=params.X_symbols, max_delta_considered=1000)

        entropies_k_forgetful.append(mean_e_forgetful)
        entropies_k_hmm.append(np.mean(results))

    plt.figure()
    plt.plot(entropies_k_forgetful, marker='.', label="Forgetful receiver")
    plt.plot(entropies_k_hmm, marker='.', label="HMM receiver")
    plt.xlabel("K")
    plt.ylabel("average estimation entropy")
    plt.legend()
    plt.grid(True)
    plt.show()