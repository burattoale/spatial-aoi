from utils.poibin import PoiBin
from numba import jit
import math
import numpy as np

from environment import NodeDistribution, DTMC
from utils import entropy, ps, prob_x_given_y_0, h_y_delta


if __name__ == "__main__":
    # PARAMETERS
    q = 5e-3
    eta = 1
    zeta = 1e-4
    epsilon = 0.1
    rho = 5e-2
    alpha = 0.02
    R = 10
    x_symbols = [0, 1]
    y_symbols = [0, 1]

    #### SIM Length ####
    sim_length = 1000


    entropies = []
    without_tx = []
    ####################
    #### SIMULATION ####
    ####################
    for K in range(4,60):
        m = math.floor(rho * np.pi * R*K)
        p_succ = ps(m, zeta, epsilon)
        rng = np.random.default_rng()
        last_y, delta = 0, 0 # initialize variables
        sim_entropies = []
        for _ in range(1):
            states = []
            mm = DTMC(q, eta)
            for i in range(sim_length):
                if i == 0:
                    x_state = mm.state
                    last_y = x_state
                    delta = 0
                    states.append((last_y, delta))
                    continue
                x_state = mm.step()
                if rng.random() <= p_succ:
                    last_y = x_state
                    delta = 0
                else:
                    delta += 1
                states.append((last_y, delta))
            states_entropy = list(set(states))
            sim_entropies.append(entropy(mm.A, mm.pi, K, R, alpha, m, zeta, epsilon, states_entropy))
        entropies.append(np.mean(sim_entropies))


    import matplotlib.pyplot as plt

    source_entropy = np.sum(- mm.pi * np.log2(mm.pi))
    for state in states:
        y, delta = state
        p_x_given_y_0 = np.array([prob_x_given_y_0(0, y, mm.pi, K, R, alpha), prob_x_given_y_0(1, y, mm.pi, K, R, alpha)])
        without_tx.append(h_y_delta(p_x_given_y_0, mm.A, delta))

    plt.plot(entropies)
    plt.xticks(np.arange(0,57, 4),np.arange(40, 601, step=40))
    plt.show()

    plt.figure()
    plt.plot(without_tx)
    plt.plot([source_entropy]*len(without_tx))
    plt.grid(True)
    plt.show()