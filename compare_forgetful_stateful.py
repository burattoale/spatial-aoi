import math
import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

from hmm_joint_prob import hmm_entropy, run_hmm_simulation
from main_simulation import run_monte_carlo_simulation
from utils import overall_entropy, SimulationParameters
from environment import DTMC, NodeDistribution

def do_simulation_pair(params:SimulationParameters, sim_length:int):
    params.beta = 0
    time_evolution, mean_e_hmm, mean_est_error, mean_est_err_short = hmm_entropy(params, sim_length)

    time_evolution_loc_aware, mean_e_hmm_loc_aware, mean_est_error_loc, mean_est_e_loc_short = hmm_entropy(params, sim_length, loc_aware=True)
    time_evolution_loc_aware_mc, mean_e_hmm_loc_aware_mc, mean_est_error_loc_mc, mean_est_e_loc_short_mc, _ = run_hmm_simulation(params, num_time_steps=sim_length, loc_aware=True)
     

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
    del time_evolution, time_evolution_forgetful, time_evolution_loc_aware, time_evolution_loc_aware_mc, _

    return mean_e_forgetful, mean_e_hmm, mean_e_hmm_loc_aware, mean_e_hmm_loc_aware_mc, mean_est_error, mean_est_error_loc, mean_est_error_loc_mc, mean_est_err_short, mean_est_e_loc_short, mean_est_e_loc_short_mc

def parallel_k(k:int, params:SimulationParameters, sim_length:int, n_topologies:int, parallel_jobs:int):
    params.K = k
    params.m = math.floor(rho * np.pi * (params.R_unit*k)**2)
    params.Y_symbols = [i for i in range(len(params.X_symbols) * k + 2)] # symbols for the location aware scenario
    #results = Parallel(n_jobs=math.ceil(n_topologies/parallel_jobs), backend="loky", verbose=1)(
    results = Parallel(n_jobs=5, backend="loky", verbose=1)(
        delayed(do_simulation_pair)(params, sim_length) for _ in range(n_topologies)
    )
    # average the results from the run
    averages = np.mean(results, axis=0)
    return averages

if __name__ == "__main__":
    sim_length = 10000
    n_topologies = 50
    parallel_jobs = 5 # how many k to do symultaneously
    k_max = 50

    q = 0.005
    eta = 1
    zeta = 0.0001
    epsilon = 0.1
    rho = 5e-2
    R = 10
    K = 5
    m = math.floor(rho * np.pi * (K*R)**2)
    alpha = 0.1
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
    res = Parallel(n_jobs=parallel_jobs, backend="loky", verbose=1)(
        delayed(parallel_k)(k, params, sim_length, n_topologies, parallel_jobs) for k in range(5, k_max, 3)
    )
    # for k in tqdm(range(1,k_max), desc="Working over Ks"):
    #     params.K = k
    #     params.m = math.floor(rho * np.pi * (params.R_unit*k)**2)
    #     results = Parallel(n_jobs=math.floor(n_topologies/parallel_jobs), backend="loky", verbose=1)(
    #         delayed(do_simulation_pair)(params, sim_length) for _ in range(n_topologies)
    #     )
    #     # average the results from the run
    #     averages = np.mean(results, axis=0)
    #     entropies_k_forgetful.append(averages[0])
    #     entropies_k_hmm.append(averages[1])

        # for the forgetful receiver create DTMC and tx probability vector
        # mm = DTMC(params.q, params.eta)
        # topology = NodeDistribution(params.rho, params.R_unit, params.K, params.alpha, params.beta)
        # mean_e_forgetful = overall_entropy(A=mm.A, pi=mm.pi, K=params.K, 
        #                                   R_unit=params.R_unit, alpha=params.alpha, m=topology.m, 
        #                                   zeta=topology.tx_probabilities, epsilon=params.epsilon, 
        #                                   states=params.X_symbols, max_delta_considered=1000)

        # entropies_k_forgetful.append(mean_e_forgetful)
        # entropies_k_hmm.append(np.mean(results))
    res = np.array(res)
    entropies_k_forgetful = res[:,0]
    entropies_k_hmm = res[:,1]
    entropies_k_hmm_loc_aware = res[:,2]
    entropies_k_hmm_loc_aware_mc = res[:,3]

    plt.figure()
    plt.semilogy(np.arange(5,k_max,3)*R, entropies_k_forgetful, marker='.', label="Forgetful receiver")
    plt.semilogy(np.arange(5,k_max,3)*R, entropies_k_hmm, marker='.', label="HMM receiver")
    plt.semilogy(np.arange(5,k_max,3)*R, entropies_k_hmm_loc_aware,linestyle=":", marker='.', label="HMM receiver loc aware")
    plt.semilogy(np.arange(5,k_max,3)*R, entropies_k_hmm_loc_aware_mc,linestyle=":", marker='.', label="HMM receiver loc aware Monte Carlo")
    plt.scatter(R+np.argmin(entropies_k_forgetful)*R, np.min(entropies_k_forgetful), marker='d', color='r', label="Minimum entropy forgetful")
    plt.scatter(R+np.argmin(entropies_k_hmm)*R, np.min(entropies_k_hmm), marker='s', color='r', label="Minimum entropy HMM")
    plt.xlabel("radius R [m]")
    plt.ylabel("average estimation entropy")
    plt.legend()
    plt.grid(True)
    plt.show()
    # plt.savefig(f"plots/comparison_eta_{eta}.png", bbox_inches="tight", pad_inches=0)

    results = {
        "K": k_max-1,
        "forgetful_receiver": entropies_k_forgetful,
        "hmm_receiver": entropies_k_hmm
    }

    plt.figure()
    est_error_no_loc = res[:, 4]
    est_error_loc = res[:, 5]
    est_error_loc_mc = res[:,6]
    est_error_short_no_loc = res[:, 7]
    est_error_short_loc = res[:, 8]
    est_error_short_loc_mc = res[:, 9]
    plt.semilogy(np.arange(5,k_max,3)*R, est_error_no_loc, marker='.', label="HMM receiver Viterbi")
    plt.semilogy(np.arange(5,k_max,3)*R, est_error_loc, marker='.', label="HMM receiver loc aware Viterbi")
    plt.semilogy(np.arange(5,k_max,3)*R, est_error_loc_mc, linestyle=":", marker='.', label="HMM receiver loc aware Monte Carlo Viterbi")
    plt.semilogy(np.arange(5,k_max,3)*R, est_error_short_no_loc, linestyle="--", marker='x', label="HMM receiver")
    plt.semilogy(np.arange(5,k_max,3)*R, est_error_short_loc, linestyle="--", marker='x', label="HMM receiver loc aware")
    plt.semilogy(np.arange(5,k_max,3)*R, est_error_short_loc_mc, linestyle=":", marker='.', label="HMM receiver loc aware Monte Carlo")
    plt.xlabel("radius R [m]")
    plt.ylabel("average estimation error probability")
    plt.legend()
    plt.grid()
    plt.show()

    # with open(f"results/comparison_entropies_eta_{eta}.pkl", 'wb') as file:
    #     pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)