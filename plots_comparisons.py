from utils import SimulationParameters
from hmm_joint_prob import run_hmm_simulation
from utils import overall_entropy

from copy import deepcopy
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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
sim_length = 10000
avg_bin = []
avg_tri = []
avg_quad = []
for k in tqdm(range(5,20)):
    params = deepcopy(initial_params)
    params.K = k
    params.m = math.floor(initial_params.rho * np.pi * (initial_params.K*initial_params.R_unit)**2)
    params_bin = deepcopy(params)
    params_tri = deepcopy(params)
    params_tri.X_symbols = [0,1,2]
    params_tri.Y_symbols = [0,1,2,3]
    params_quad = deepcopy(params)
    params_quad.X_symbols = [0,1,2,3,4]
    params_quad.Y_symbols = [0,1,2,3,4,5]
    temp_bin = []
    temp_tri = []
    temp_quad = []
    for i in range(10):
        _, avg_entropy_bin, _, _, _ = run_hmm_simulation(params_bin, sim_length, seed=i, non_binary=False)
        _, avg_entropy_tri, _, _, _ = run_hmm_simulation(params_tri, sim_length, seed=i, non_binary=True)
        _, avg_entropy_quad, _, _, _ = run_hmm_simulation(params_quad, sim_length, seed=i, non_binary=True)
        temp_bin.append(avg_entropy_bin)
        temp_tri.append(avg_entropy_tri)
        temp_quad.append(avg_entropy_quad)
    avg_bin.append(np.mean(temp_bin))
    avg_tri.append(np.mean(temp_tri))
    avg_quad.append(np.mean(temp_quad))


plt.figure(1)
xs = np.arange(5,20)
plt.plot(xs, avg_bin, marker="x", label = "HMM binary source")
plt.plot(xs, avg_tri, marker=".", label = "HMM ternaryary source")
plt.plot(xs, avg_quad, marker="d", label = "HMM 5-symbols source")
plt.xlabel("Number of zones (K)")
plt.ylabel("Average entropy")
plt.grid(True)
plt.legend()
plt.show()
