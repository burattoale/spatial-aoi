import numpy as np
import math
import matplotlib.pyplot as plt

from hmm_joint_prob import hmm_entropy
from utils import SimulationParameters
from environment import HiddenMM

if __name__ == "__main__":
    sim_length = 50000
     # sim parameters
    q = 0.01
    eta = 1
    zeta = 0.002
    epsilon = 0
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

    entropies, _ = hmm_entropy(params, simulation_length=sim_length)
    hmm = HiddenMM(params)

    plt.figure()
    source_entropy = - hmm.pi[0] * np.log2(hmm.pi[0]) - hmm.pi[1] * np.log2(hmm.pi[1])
    plt.plot(np.arange(sim_length), entropies, label=r"$H(X_n \mid Y^n=y^n)$")
    plt.plot(np.arange(sim_length), np.ones(sim_length) * source_entropy, linestyle="-.", label="$H(X)$")
    plt.grid(True)
    plt.legend()
    plt.savefig("plots/sequence_entropy.png", bbox_inches="tight", pad_inches=0)
    plt.close()

    mean_entropies_symmetric, mean_entropies_asymmetric = [], []
    for m in range(1, 40):
        print(f"Simulation for m = {m}")
        params.eta = 1
        params.m = m
        params.zeta = 1/m
        _, mean_entropy_symmetric = hmm_entropy(params, simulation_length=sim_length)
        mean_entropies_symmetric.append(mean_entropy_symmetric)
        params.eta = 10
        _, mean_entropy_asymmetric = hmm_entropy(params, simulation_length=sim_length)
        mean_entropies_asymmetric.append(mean_entropy_asymmetric)

    plt.figure()
    plt.semilogy(np.arange(1,40), mean_entropies_symmetric, label="entropy symmetric source")
    plt.semilogy(np.arange(1,40), mean_entropies_asymmetric, label="entropy asymmetric source")
    plt.xlabel(r"number of sources, $M$")
    plt.ylabel("average state estimation entropy")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/entropy_over_M.png", bbox_inches="tight", pad_inches=0)
