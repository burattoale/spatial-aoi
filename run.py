from utils.poibin import PoiBin
from numba import jit
import math
import numpy as np
import matplotlib.pyplot as plt

from utils import SimulationParameters
from hmm_joint_prob import run_hmm_simulation
from environment import HiddenMM


if __name__ == "__main__":
    sim_length = 10000
     # sim parameters
    q = 0.01
    eta = 1
    zeta = 1e-4
    epsilon = 0
    rho = 5e-2
    R = 10
    K = 5
    m = math.floor(rho * np.pi * (K*R)**2)
    alpha = 0.02
    x_symbols = [0, 1, 2,3,4]
    y_symbols = [0, 1, 2, 3,4,5]
    y_translation = {
        0:"0",
        1:"1",
        2:"C",
        3:"I"
    }

    params = SimulationParameters(q, eta, zeta, epsilon, m=m, K=K, alpha=alpha, R_unit=R, X_symbols=x_symbols, Y_symbols=y_symbols)

    entropies, _, _, _, Y = run_hmm_simulation(params, sim_length, seed=0, non_binary=True)
    hmm = HiddenMM(params)

    print(Y[3990:4050])

    plt.figure()
    source_entropy = - np.sum(hmm.pi * np.log2(hmm.pi))
    plt.plot(np.arange(sim_length), entropies, label=r"$H(X_n \mid Y^n=y^n)$")
    plt.plot(np.arange(sim_length), np.ones(sim_length) * source_entropy, linestyle="-.", label="$H(X)$")
    plt.grid(True)
    plt.legend()
    plt.show()
    plt.savefig("plots/sequence_entropy_5_source_states.png", bbox_inches="tight", pad_inches=0)
    plt.close()

    avg_entropies = []
    for k in range(1, 20):
        params.K = k
        params.m = math.floor(rho * np.pi * (K*R)**2)
        _, avg_entropy, _, _, Y = run_hmm_simulation(params, sim_length, seed=0, non_binary=True)
        avg_entropies.append(avg_entropy)

    plt.figure()
    plt.plot(np.arange(1, 20), avg_entropies, label=r"$ average H(X_n \mid Y^n=y^n)$")
    plt.grid(True)
    plt.legend()
    plt.xlabel("number of zones (K)")
    plt.ylabel("average estimation entropy")
    plt.show()
    plt.savefig("plots/average_entropy_5_source_states.png", bbox_inches="tight", pad_inches=0)
    plt.close()