import numpy as np
import math
import matplotlib.pyplot as plt

from hmm_joint_prob import hmm_entropy
from utils import SimulationParameters
from environment import HiddenMM
from utils import run_monte_carlo_simulation, run_monte_carlo_simulation_forced_sequence

if __name__ == "__main__":
    sim_length = 10000
     # sim parameters
    q = 0.005
    eta = 5
    zeta = 1e-4
    epsilon = 0
    rho = 5e-2
    R = 10
    K = 5
    m = math.floor(rho * np.pi * (R*K)**2)
    alpha = 0.02
    x_symbols = [0, 1]
    y_symbols = [0, 1, 2]
    y_translation = {
        0:"0",
        1:"1",
        2:"C",
        3:"I"
    }
    # frazione messaggi che dicono 0 e 1
    # evoluzione temporale legata ai messaggi

    params = SimulationParameters(q, eta, zeta, epsilon, rho=rho, m=m, K=K, alpha=alpha, R_unit=R, X_symbols=x_symbols, Y_symbols=y_symbols)

    etas = [1, 5, 20]
    Ks = [5, 30]
    for eta in etas:
        for K in Ks:
            print(f"\n--- Simulating for eta={eta}, K={K} ---")
            params.eta = eta
            params.K = K
            params.m = math.floor(rho * np.pi * (R * K)**2)
            entropies, _, _, _, (Y, X_true, cond_probs_x1, cond_probs_x), (cdf_h, gamma_vec) = hmm_entropy(params, simulation_length=sim_length, seed=0)
            # Calcola la frazione di messaggi che dicono 0 e 1
            Y = np.array(Y)
            total = len(Y)
            frac_0 = np.sum(Y == 0) / total
            frac_1 = np.sum(Y == 1) / total
            print(f"Frazione messaggi che dicono 0: {frac_0:.4f}")
            print(f"Frazione messaggi che dicono 1: {frac_1:.4f}")
            # Calcola la frazione di messaggi che dicono 0 correttamente e 1 correttamente
            X_true = np.array(X_true)
            correct_0 = np.sum((Y == 0) & (X_true == 0)) / total
            correct_1 = np.sum((Y == 1) & (X_true == 1)) / total
            print(f"Frazione messaggi che dicono 0 correttamente: {correct_0:.4f}")
            print(f"Frazione messaggi che dicono 1 correttamente: {correct_1:.4f}")
            hmm = HiddenMM(params, np.array([params.zeta]*params.K))
            source_entropy = - np.sum(hmm.pi * np.log2(hmm.pi))

            # Trova i time step in cui c'è mismatch tra Y e X_true
            mismatch_0to1 = np.where((X_true == 0) & (Y == 1))[0]  # vero 0, ricevuto 1
            mismatch_1to0 = np.where((X_true == 1) & (Y == 0))[0]  # vero 1, ricevuto 0

            # Trova i time step in cui c'è match corretto tra Y e X_true
            match_0 = np.where((X_true == 0) & (Y == 0))[0]  # vero 0, ricevuto 0
            match_1 = np.where((X_true == 1) & (Y == 1))[0]  # vero 1, ricevuto 1

            # Plotta l'entropia e indica i mismatch e match con colori diversi
            plt.figure()
            plt.plot(np.arange(sim_length), entropies, label=r"$H(X_n \mid Y^n=y^n)$")
            plt.plot(np.arange(sim_length), np.ones(sim_length) * source_entropy, linestyle="-.", label=r"$H(X)$")

            # Find the index where the switch in Y from 0 to 1 happens around 4949
            switch_idx = None
            for i in range(4945, 4955):
                if Y[i-1] == 2 and Y[i] == 1:
                    switch_idx = i
                    break

            if switch_idx is not None:
                print("cond_probs_x before switch (Y=0):", cond_probs_x[switch_idx-1])
                print("cond_probs_x at switch (Y=1):", cond_probs_x[switch_idx])
            else:
                print("No switch from Y=0 to Y=1 found in the specified range.")
            
            #shading_start = None
            #label_added = False
            #for t in range(sim_length):
            #    if shading_start is None:
            #        if Y[t] == 1 and cond_probs_x1[t] > 0.5:
            #            shading_start = t
            #    elif cond_probs_x1[t] < 0.5:
            #        if not label_added:
            #            plt.axvspan(shading_start, t, color='green', alpha=0.3, label=r'$p(X_n=1|Y^n) > 0.5$ after $Y_n=1$')
            #            label_added = True
            #        else:
            #            plt.axvspan(shading_start, t, color='green', alpha=0.3)
            #        shading_start = None
            #if shading_start is not None:
            #    if not label_added:
            #        plt.axvspan(shading_start, sim_length, color='green', alpha=0.3, label=r'$p(X_n=1|Y^n) > 0.5$ after $Y_n=1$')
            #    else:
            #        plt.axvspan(shading_start, sim_length, color='green', alpha=0.3)

            #plt.scatter(mismatch_0to1, entropies[mismatch_0to1], color='red', s=10, label="Vero 0, ricevuto 1", zorder=5)
            #plt.scatter(mismatch_1to0, entropies[mismatch_1to0], color='black', s=10, label="Vero 1, ricevuto 0", zorder=5)
            #plt.scatter(match_0, entropies[match_0], color='green', s=10, label="Vero 0, ricevuto 0", zorder=5)
            #plt.scatter(match_1, entropies[match_1], color='purple', s=10, label="Vero 1, ricevuto 1", zorder=5)
            plt.grid(True)
            plt.legend()
            plt.xlim(4900, 5150)
            plt.ylim(0, 1)
            plt.xlabel(r"time step $n$")
            plt.ylabel(r"entropy $H(X_n \mid Y^n=y^n)$")
            plt.title(f"Entropia e match/mismatch per eta={eta}, K={K}")
            plt.savefig(f"plots/entropy_mismatch_eta{eta}_K{K}.png", bbox_inches="tight", pad_inches=0)

            # Simulate forgetful receiver with FORCED sequence from HMM
            # Use the exact same Y sequence that the HMM receiver saw
            _, entropies_forgetful, (Y_forgetful, X_true_forgetful) = run_monte_carlo_simulation_forced_sequence(
                params, Y, X_true, num_burn_in_steps=0)

            # The Y_forgetful and X_true_forgetful are just the same as the input sequences
            # since we're forcing them, but we keep them for consistency
            Y_forgetful = np.array(Y_forgetful)
            X_true_forgetful = np.array(X_true_forgetful)
            mismatch_0to1_f = np.where((X_true_forgetful == 0) & (Y_forgetful == 1))[0]
            mismatch_1to0_f = np.where((X_true_forgetful == 1) & (Y_forgetful == 0))[0]
            match_0_f = np.where((X_true_forgetful == 0) & (Y_forgetful == 0))[0]
            match_1_f = np.where((X_true_forgetful == 1) & (Y_forgetful == 1))[0]

            plt.figure()
            plt.plot(np.arange(sim_length), entropies_forgetful, label=r"Forgetful $H(X_n \mid Y^n=y^n)$")
            plt.plot(np.arange(sim_length), np.ones(sim_length) * source_entropy, linestyle="-.", label=r"$H(X)$")
            #plt.scatter(mismatch_0to1_f, entropies_forgetful[mismatch_0to1_f], color='red', s=10, label="Vero 0, ricevuto 1", zorder=5)
            #plt.scatter(mismatch_1to0_f, entropies_forgetful[mismatch_1to0_f], color='black', s=10, label="Vero 1, ricevuto 0", zorder=5)
            #plt.scatter(match_0_f, entropies_forgetful[match_0_f], color='green', s=10, label="Vero 0, ricevuto 0", zorder=5)
            #plt.scatter(match_1_f, entropies_forgetful[match_1_f], color='purple', s=10, label="Vero 1, ricevuto 1", zorder=5)
            plt.grid(True)
            plt.legend()
            plt.xlim(4900, 5150)
            plt.ylim(0, 1)
            plt.xlabel(r"time step $n$")
            plt.ylabel(r"entropy $H(X_n \mid Y^n=y^n)$")
            plt.title(f"Forgetful receiver (forced sequence): Entropia e match/mismatch per eta={eta}, K={K}")
            plt.savefig(f"plots/entropy_mismatch_forgetful_eta{eta}_K{K}.png", bbox_inches="tight", pad_inches=0)

        # Save traces for TikZ plots
            # Save traces for TikZ plots, using integer formatting for integer arrays
            np.savetxt(
                f"plots/tikz_entropy_hmm_eta{eta}_K{K}.txt",
                np.column_stack((np.arange(sim_length), entropies)),
                delimiter=" ",
                header="n entropy",
                comments="",
                fmt=["%d", "%.8f"]
            )
            np.savetxt(
                f"plots/tikz_entropy_forgetful_eta{eta}_K{K}.txt",
                np.column_stack((np.arange(sim_length), entropies_forgetful)),
                delimiter=" ",
                header="n entropy_forgetful",
                comments="",
                fmt=["%d", "%.8f"]
            )
            np.savetxt(
                f"plots/tikz_reception_pattern_eta{eta}_K{K}.txt",
                np.column_stack((np.arange(sim_length), Y)),
                delimiter=" ",
                header="n Y",
                comments="",
                fmt=["%d", "%d"]
            )
            np.savetxt(
                f"plots/tikz_source_entropy_eta{eta}_K{K}.txt",
                np.column_stack((np.arange(sim_length), np.ones(sim_length) * source_entropy)),
                delimiter=" ",
                header="n source_entropy",
                comments="",
                fmt=["%d", "%.8f"]
            )

        

    mean_entropies_symmetric, mean_entropies_asymmetric = [], []
    for k in range(1, 10):
        print(f"Simulation for k = {k}")
        params.eta = 1
        params.K = k
        params.m = math.floor(rho * np.pi * (R * k)**2)
        _, mean_entropy_symmetric, _, _, _, _ = hmm_entropy(params, simulation_length=sim_length)
        mean_entropies_symmetric.append(mean_entropy_symmetric)
        params.eta = 5
        _, mean_entropy_asymmetric, _, _, _, _ = hmm_entropy(params, simulation_length=sim_length)
        mean_entropies_asymmetric.append(mean_entropy_asymmetric)

    plt.figure()
    plt.semilogy(np.arange(1,10)*R, mean_entropies_symmetric, label=r"entropy $\eta=1$")
    plt.semilogy(np.arange(1,10)*R, mean_entropies_asymmetric, label=r"entropy $\eta=5$")
    plt.xlabel(r"coverage radius $R_m$")
    plt.ylabel("average state estimation entropy")
    plt.legend()
    plt.grid(True)
    plt.show()
    #plt.savefig("plots/entropy_over_M.png", bbox_inches="tight", pad_inches=0)
