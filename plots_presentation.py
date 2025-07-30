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
with open("results/binary_simulations_eta_R_unit_5.pkl", 'rb') as f:
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
plt.show()


# Define the x values
xs = np.arange(1, 50)

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
with open("results/binary_simulations.pkl", 'rb') as f:
    data = pickle.load(f)
R_unit = 10
forgetful = data["forgetful"]
hmm = data["hmm"]

xs = np.arange(1,50) * R_unit
values_forget = np.array(list(forgetful.values()))
values_hmm = np.array(list(hmm.values()))
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