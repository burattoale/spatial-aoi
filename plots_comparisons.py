from utils import SimulationParameters
from hmm_joint_prob import run_hmm_simulation
from utils import overall_entropy

from copy import deepcopy
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

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
        R_unit=10,
        Y_symbols=[0,1,2]
    )
sim_length = 20000
avg_bin = []
avg_tri = []
avg_quad = []
avg_bin_loc = []
avg_tri_loc = []
avg_quad_loc = []
for k in tqdm(range(5,20)):
    params = deepcopy(initial_params)
    params.K = k
    params.m = math.floor(initial_params.rho * np.pi * (initial_params.K*initial_params.R_unit)**2)
    params_bin = deepcopy(params)
    params_tri = deepcopy(params)
    params_tri.alpha = 0.06
    params_quad = deepcopy(params)
    params_quad.X_symbols = [0,1,2,3,4]
    params_quad.Y_symbols = [0,1,2,3,4,5]
    temp_bin = []
    temp_bin_loc = []
    temp_tri = []
    temp_tri_loc = []
    temp_quad = []
    temp_quad_loc = []
    for i in range(30):
        _, avg_entropy_bin, _, _, _ = run_hmm_simulation(params_bin, sim_length, seed=i, non_binary=False)
        params_bin_loc = deepcopy(params_bin)
        params_bin_loc.Y_symbols = [i for i in range(params_bin.K * 2 +1)]
        _, avg_entropy_bin_loc, _, _, _ = run_hmm_simulation(params_bin_loc, sim_length, seed=i, non_binary=False, loc_aware=True)
        _, avg_entropy_tri, _, _, _ = run_hmm_simulation(params_tri, sim_length, seed=i, non_binary=False)
        params_tri_loc = deepcopy(params_tri)
        params_tri_loc.Y_symbols = [i for i in range(params_bin.K * 2 +1)]
        _, avg_entropy_tri_loc, _, _, _ = run_hmm_simulation(params_tri_loc, sim_length, seed=i, non_binary=False, loc_aware=True)
        params_quad = deepcopy(params_tri)
        params_quad.alpha = 0.1
        params_quad_loc = deepcopy(params_tri_loc)
        params_quad_loc.alpha = 0.1
        _, avg_entropy_quad, _, _, _ = run_hmm_simulation(params_quad, sim_length, seed=i, non_binary=False)
        _, avg_entropy_quad_loc, _, _, _ = run_hmm_simulation(params_quad_loc, sim_length, seed=i, non_binary=False, loc_aware=True)
        #_, avg_entropy_quad, _, _, _ = run_hmm_simulation(params_quad, sim_length, seed=i, non_binary=True)
        temp_bin.append(avg_entropy_bin)
        temp_bin_loc.append(avg_entropy_bin_loc)
        temp_tri.append(avg_entropy_tri)
        temp_tri_loc.append(avg_entropy_tri_loc)
        temp_quad.append(avg_entropy_quad)
        temp_quad_loc.append(avg_entropy_quad_loc)
    avg_bin.append(np.mean(temp_bin))
    avg_bin_loc.append(np.mean(temp_bin_loc))
    avg_tri.append(np.mean(temp_tri))  
    avg_tri_loc.append(np.mean(temp_tri_loc))  
    avg_quad.append(np.mean(temp_quad))
    avg_quad_loc.append(np.mean(temp_quad_loc))


plt.figure(1)
xs = np.arange(5,20)
plt.plot(xs, avg_bin, marker="x", label = r"HMM ,$\alpha=0.02$")
plt.plot(xs, avg_bin_loc, marker="x", label = r"HMM ,$\alpha=0.02$ loc. aware")
plt.plot(xs, avg_tri, marker=".", label = r"HMM ,$\alpha=0.06$")
plt.plot(xs, avg_tri_loc, marker=".", label = r"HMM ,$\alpha=0.06$ loc. aware")
plt.plot(xs, avg_quad, marker=".", label = r"HMM ,$\alpha=0.1$")
plt.plot(xs, avg_quad_loc, marker=".", label = r"HMM ,$\alpha=0.1$ loc. aware")
#plt.plot(xs, avg_quad, marker="d", label = "HMM 5-symbols source")
plt.xlabel("Number of zones (K)")
plt.ylabel("Average estimation entropy")
plt.grid(True)
plt.legend()
plt.show()


results = {
    "vanilla": {
        0.02: avg_bin,
        0.06: avg_tri,
        0.1: avg_quad
    },
    "loc_aware": {
        0.02: avg_bin_loc,
        0.06: avg_tri_loc,
        0.1: avg_quad_loc
    }
}
with open("results/alpha_loc_aware.pkl", 'wb') as f:
    pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
