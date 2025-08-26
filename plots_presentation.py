import pickle
import numpy as np
import matplotlib.pyplot as plt
from copy import copy

colors = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan"
]
R_unit = 5
with open("results/binary_source.pkl", 'rb') as f:
    data_eta = pickle.load(f)

# plot the evolution with gowing number of sections
plt.figure(0)
eta_1 = data_eta["forgetful"]["eta:1"]
eta_5 = data_eta["forgetful"]["eta:5"]
eta_9 = data_eta["forgetful"]["eta:9"]

hmm_eta_1 = data_eta["hmm"]["eta:1"]
hmm_eta_5 = data_eta["hmm"]["eta:5"]
hmm_eta_9 = data_eta["hmm"]["eta:9"]

plt.semilogy(np.array(list(eta_1.keys())) * R_unit,eta_1.values(), color=colors[0], marker='.', markevery=4, label=r"$\eta=1$, Forgetful")
plt.semilogy(np.array(list(eta_5.keys())) * R_unit,eta_5.values(), color=colors[1], marker='.', markevery=4, label=r"$\eta=5$, Forgetful")
plt.semilogy(np.array(list(eta_9.keys())) * R_unit,eta_9.values(), color=colors[2], marker='.', markevery=4, label=r"$\eta=9$, Forgetful")
plt.semilogy(np.array(list(hmm_eta_1.keys())) * R_unit,hmm_eta_1.values(), color=colors[0], linestyle='--', marker='x', markevery=4, label=r"$\eta=1$, HMM")
plt.semilogy(np.array(list(hmm_eta_5.keys())) * R_unit,hmm_eta_5.values(), color=colors[1], linestyle='--', marker='x', markevery=4, label=r"$\eta=5$, HMM")
plt.semilogy(np.array(list(hmm_eta_9.keys())) * R_unit, hmm_eta_9.values(), color=colors[2], linestyle='--', marker='x', markevery=4, label=r"$\eta=9$, HMM")

plt.grid(which='both', color='grey', linestyle=':', linewidth=0.7)
plt.xlabel(r"Coverage radius ($R$)")
plt.ylabel(r"Average estimation entropy")
plt.legend(ncols=2)
plt.subplots_adjust(left=0.15)
plt.xlim(left=1)
# Plot for eta:1 with different number of sources
plt.figure(100)
source_counts = ["binary", "three", "five", "seven", "ten"]  # Example source counts, adjust as needed
for idx, n_sources in enumerate(source_counts):
    with open(f"results/{n_sources}_source.pkl", 'rb') as f:
        data_multi = pickle.load(f)
    hmm_eta_1_multi = data_multi["hmm"]["eta:1"]
    plt.semilogy(np.array(list(hmm_eta_1_multi.keys())) * R_unit, hmm_eta_1_multi.values(),
                 color=colors[idx], marker='x', markevery=4, label=f"{n_sources} sources, HMM")
    # Location-aware comparison
    try:
        with open(f"results/{n_sources}_source_loc_aware.pkl", 'rb') as f_loc:
            data_multi_loc = pickle.load(f_loc)
        hmm_eta_1_multi_loc = data_multi_loc["hmm"]["eta:1"]
        plt.semilogy(np.array(list(hmm_eta_1_multi_loc.keys())) * R_unit, hmm_eta_1_multi_loc.values(),
                     color=colors[idx], linestyle='--', marker='o', markevery=4, label=f"{n_sources} sources, HMM (Loc-Aware)")
    except FileNotFoundError:
        pass  # Skip if location-aware data is not available
plt.grid(which='both', color='grey', linestyle=':', linewidth=0.7)
plt.xlabel(r"Coverage radius ($R$)")
plt.ylabel(r"Average estimation entropy")
plt.legend()
plt.xlim(left=1)
plt.show()

# Plot for location-aware data (same as first plot but with location-aware results)
with open("results/binary_source_loc_aware.pkl", 'rb') as f:
    data_eta_loc = pickle.load(f)

plt.figure(4)
# Read location-aware data for different alpha values
with open("results/binary_source_alpha.pkl", 'rb') as f_alpha:
    data_eta_alpha = pickle.load(f_alpha)
with open("results/binary_source_loc_aware_alpha.pkl", 'rb') as f_alpha_loc:
    data_eta_loc_alpha = pickle.load(f_alpha_loc)

alpha_keys = ["alpha:0.02", "alpha:0.05", "alpha:0.1"]
colors_alpha = colors[:len(alpha_keys)]

for idx, alpha_key in enumerate(alpha_keys):
    eta_alpha = data_eta_alpha["hmm"][alpha_key]
    eta_loc_alpha = data_eta_loc_alpha["hmm"][alpha_key]
    plt.semilogy(np.array(list(eta_alpha.keys())) * R_unit, eta_alpha.values(),
                 color=colors_alpha[idx], marker='.', markevery=4, label=fr"${alpha_key}$, HMM")
    plt.semilogy(np.array(list(eta_loc_alpha.keys())) * R_unit, eta_loc_alpha.values(),
                 color=colors_alpha[idx], linestyle='--', marker='x', markevery=4, label=fr"${alpha_key}$, HMM (Loc-Aware)")

plt.grid(which='both', color='grey', linestyle=':', linewidth=0.7)
plt.xlabel(r"Coverage radius ($R$)")
plt.ylabel(r"Average estimation entropy")
plt.legend(ncols=2)
plt.subplots_adjust(left=0.15)
plt.xlim(left=1)
plt.show()

plt.figure(10)
err_eta_1 = data_eta["hmm_err"]["eta:1"]
err_eta_5 = data_eta["hmm_err"]["eta:5"]
err_eta_9 = data_eta["hmm_err"]["eta:9"]
err_eta_25 = data_eta["hmm_err"]["eta:25"]

plt.semilogy(np.array(list(err_eta_1.keys())) * R_unit, err_eta_1.values(), color=colors[0], marker='.', markevery=4, label=r"$\eta=1$, HMM")
plt.semilogy(np.array(list(err_eta_5.keys())) * R_unit, err_eta_5.values(), color=colors[1], marker='d', markevery=4, label=r"$\eta=5$, HMM")
plt.semilogy(np.array(list(err_eta_9.keys())) * R_unit, err_eta_9.values(), color=colors[2], marker='*', markevery=4, label=r"$\eta=9$, HMM")
plt.semilogy(np.array(list(err_eta_25.keys())) * R_unit, err_eta_25.values(), color=colors[3], marker='x', markevery=4, label=r"$\eta=25$, HMM")
plt.grid(which='both', color='grey', linestyle=':', linewidth=0.7)
plt.legend()
plt.xlabel(r"Coverage radius ($R$)")
plt.ylabel(r"Average estimation error")
plt.show()


# Define the x values
xs = [1,5,9,25]#np.arange(1, 50)

# Extract the relevant data for forgetful and HMM

forgetful = data_eta["forgetful"]
forgetful_optimal_R = [np.argmin(list(forgetful[key].values())) for key in forgetful.keys()]

forgetful_optimal_H = [np.min(list(forgetful[key].values())) for key in forgetful.keys()]

hmm = data_eta["hmm"]
hmm_optimal_R = [np.argmin(list(hmm[key].values())) for key in hmm.keys()]

hmm_optimal_H = [np.min(list(hmm[key].values())) for key in hmm.keys()]


# Plot the data
plt.figure(1)
plt.plot(xs, np.array(forgetful_optimal_R)* R_unit, marker='.', label="Forgetful")
plt.plot(xs, np.array(hmm_optimal_R)* R_unit, marker='.', label="HMM")
plt.xlabel(r"Asymmetry coefficient ($\eta$)")
plt.ylabel(r"Optimal coverage radius ($R^*$)")
plt.xlim(left=1)
plt.legend()
plt.grid(which='both', color='grey', linestyle=':', linewidth=0.7)
plt.show()

# plot optimal entropy
plt.figure(2)
plt.plot(xs, forgetful_optimal_H, marker='.', label="Forgetful")
plt.plot(xs, hmm_optimal_H, marker='.', label="HMM")
plt.xlabel(r"Asymmetry coefficient ($\eta$)")
plt.ylabel(r"Minimum entropy ($H^*$)")
plt.grid(which='both', color='grey', linestyle=':', linewidth=0.7)
plt.legend()
plt.xlim(left=1)
plt.show()

# same as first plot but use a dashed line 
# when minimum is achieved for both hmm and forgetful (this is the optimizetd tx probability)
plt.figure(3)
with open("results/binary_source.pkl", 'rb') as f:
    data = pickle.load(f)
R_unit = 10
forgetful = data["forgetful"]
hmm = data["hmm"]

xs = np.arange(10,50) * R_unit
values_forget = np.array(list(forgetful["eta:1"].values()))
values_hmm = np.array(list(hmm["eta:1"].values()))
min_forget = np.min(values_forget)
idx_forget = np.argmin(values_forget)
min_hmm = np.min(values_hmm)
idx_hmm = np.argmin(values_hmm)
plt.semilogy(xs, values_forget, color=colors[0], marker='.', markevery=4, label="Forgetful")
plt.semilogy(xs, values_hmm, color=colors[1], marker='.', markevery=4, label="HMM")
# straight lines
plt.semilogy(xs[idx_forget:], [min_forget]*len(xs[idx_forget:]), color=colors[0], linestyle=':', label=r"Forgetful, optimal $\zeta^*$")
plt.semilogy(xs[idx_hmm:], [min_hmm]*len(xs[idx_hmm:]), color=colors[1], linestyle=':', label=r"HMM, optimal $\zeta^*$")
# Configure the grid and labels
plt.grid(which='both', color='grey', linestyle=':', linewidth=0.7)
plt.xlabel(r"Coverage radius ($R$)")
plt.ylabel(r"Average estimation entropy")
plt.legend(ncols=2)
plt.xlim(left=1)

# Show the plot
plt.show()