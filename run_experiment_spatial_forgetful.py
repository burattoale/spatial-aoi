import numpy as np
import math
import matplotlib.pyplot as plt
from copy import deepcopy

from utils import SimulationParameters
from utils import run_monte_carlo_simulation_spatial, run_monte_carlo_simulation

if __name__ == "__main__":
    initial_params = SimulationParameters(
        q=0.005,
        eta=1,
        zeta=1e-4, # Example: list of base probabilities. NodeDistribution will use this.
        epsilon=0.1,
        rho=0.05,
        m_override=None,
        K=10, # Initial K for the first plot
        alpha=0.02,
        beta=0, # Default beta, will be overridden in the loop
        R_unit=10
    )
    initial_params.m = math.floor(initial_params.rho * np.pi * (initial_params.K*initial_params.R_unit)**2)

    avg_entropy_forget, evolution_forget = run_monte_carlo_simulation_spatial(initial_params, 1000, 100, seed=0, discard_logic=True)
    avg_entropy_not_forget, evolution_not_forget = run_monte_carlo_simulation_spatial(initial_params, 1000, 100, seed=0)
    print(f"Average without selective dropping: {avg_entropy_not_forget}")
    print(f"Average with selective dropping: {avg_entropy_forget}")

    plt.figure()
    plt.plot(evolution_forget, label="Evolution with drop")
    plt.plot(evolution_not_forget, label="Evolution without dropping")

    plt.xlabel("Number of zones (K)")
    plt.ylabel("Estimation entropy")

    plt.legend()
    plt.grid(True)
    plt.show()

    discarding_spatial = []
    forgetful_spatial = []
    forgetful = []
    mink = 2
    maxk = 35
    for k in range(mink, maxk):
        params = deepcopy(initial_params)
        params.K = k
        params.m = math.floor(params.rho * np.pi * (params.K*params.R_unit)**2)
        avg_discard, _ = run_monte_carlo_simulation_spatial(params, 10000, 100, seed=0, discard_logic=True)
        avg_spatial, _ = run_monte_carlo_simulation_spatial(params, 10000, 100, seed=0)
        avg_forget, _ = run_monte_carlo_simulation(params, 10000, 100, seed=0)
        discarding_spatial.append(avg_discard)
        forgetful_spatial.append(avg_spatial)
        forgetful.append(avg_forget)

    plt.figure()
    x_list = np.arange(mink, maxk)
    plt.semilogy(x_list, discarding_spatial, label="Spatial forgetful discard")
    plt.semilogy(x_list, forgetful_spatial, label="Spatial forgetful not discard")
    plt.semilogy(x_list, forgetful, label="Forgetful")

    plt.xlabel("Number of zones (K)")
    plt.ylabel("Average estimation entropy")
    plt.legend()
    plt.grid(True, which="both")
    plt.show()


    

