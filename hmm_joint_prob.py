import numpy as np
from matplotlib import pyplot as plt
import math
import copy
from tqdm import tqdm

from environment import HiddenMM, SpatialHMM
from utils import SimulationParameters
from utils import cond_prob_y_given_x, sequence_entropy

def hmm_entropy(params:SimulationParameters, simulation_length:int=10000, loc_aware=False):
    local_params:SimulationParameters = copy.deepcopy(params)
    if not loc_aware:
        local_params.Y_symbols = [0, 1, 2, 3]
        hmm = HiddenMM(local_params)
    else:
        hmm = SpatialHMM(params)
    Y = np.empty(simulation_length, dtype=object) # observations
    alpha_vec = np.empty(2) # forward variables
    cs = np.zeros(simulation_length) # scaling coefficients
    entropies = np.zeros(simulation_length)

    print(f"Starting simulation with {simulation_length} steps")
    print(f"Number of sensors: {local_params.m}")
    # initialize the first values of the simulation
    Y[0] = hmm.hidden_state
    cond_prob0 = hmm.B[0, Y[0]]
    cond_prob1 = hmm.B[1, Y[0]]
    alpha_vec[0] = hmm.pi[0] * cond_prob0
    alpha_vec[1] = hmm.pi[1] * cond_prob1
    cs[0] = 1

    for i in tqdm(range(1, simulation_length)):
        Y[i] = hmm.step()

        temp0, temp1 = 0, 0
        for id in range(2): # only two states
            temp0 += alpha_vec[id] * hmm.A[id, 0]
            temp1 += alpha_vec[id] * hmm.A[id, 1]
        alpha_vec[0] = temp0 * hmm.B[0, Y[i]]
        alpha_vec[1] = temp1 * hmm.B[1, Y[i]]
        cs[i] = np.sum(alpha_vec)
        alpha_vec /= cs[i] # normalize the forward probabilities

        p_0_y_norm = alpha_vec[0] / np.sum(alpha_vec)
        p_1_y_norm = alpha_vec[1] / np.sum(alpha_vec)
        entropies[i] = - p_0_y_norm * np.log2(p_0_y_norm+1e-12) - p_1_y_norm * np.log2(p_1_y_norm+1e-12)

    return entropies, np.mean(entropies)


if __name__ == "__main__":
    sim_length = 100
    # sim parameters
    q = 0.005
    eta = 1
    zeta = 0.0001
    epsilon = 0.1
    rho = 5e-3
    R = 10
    K = 5
    m = math.floor(rho * np.pi * (R*K)**2)
    print(f"Number of sensors: {m}")
    alpha = 0.02
    x_symbols = [0, 1]
    # y_symbols = [0, 1, 2, 3]
    # spatial HMM
    y_symbols = [i for i in range(len(x_symbols) * K + 2)]
    y_translation = {
        0:"0",
        1:"1",
        2:"C",
        3:"I"
    }

    local_params = SimulationParameters(q, eta, zeta, epsilon, m=m, K=K, alpha=alpha, R_unit=R, X_symbols=x_symbols, Y_symbols=y_symbols)

    entropies, _ = hmm_entropy(local_params, sim_length)

    #lambda_n = np.log(alpha_vec[0]/alpha_vec[1])
    # print(f"Entropy alpha method: {sequence_entropy(lambda_n)}")
    #print(f"Entropy Liva method: {sequence_entropy(lambda_other)}")


    plt.figure()
    #source_entropy = - hmm.pi[0] * np.log2(hmm.pi[0]) - hmm.pi[1] * np.log2(hmm.pi[1])
    plt.plot(np.arange(sim_length), entropies, label=r"$H(X_n \mid Y^n=y^n)$")
    #plt.plot(np.arange(sim_length), np.ones(sim_length) * source_entropy, linestyle="-.", label="$H(X)$")
    plt.grid(True)
    plt.legend()
    plt.savefig("plots/sequence_entropy.png", bbox_inches="tight", pad_inches=0)
    plt.show()



