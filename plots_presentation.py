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

with open("results/binary_simulations_eta.pkl", 'rb') as f:
    data_eta = pickle.load(f)

# plot the evolution with gowing number of sections
plt.figure(0)
eta_1 = data_eta["forgetful"]["eta:1"]
eta_5 = data_eta["forgetful"]["eta:5"]
eta_9 = data_eta["forgetful"]["eta:9"]

hmm_eta_1 = data_eta["hmm"]["eta:1"]
hmm_eta_5 = data_eta["hmm"]["eta:5"]
hmm_eta_9 = data_eta["hmm"]["eta:9"]

plt.semilogy(np.array(list(eta_1.keys())) * 10,eta_1.values(), color=colors[0], marker='.', markevery=4, label=r"$\eta=1$, Forgetful")
plt.semilogy(np.array(list(eta_5.keys())) * 10,eta_5.values(), color=colors[1], marker='.', markevery=4, label=r"$\eta=5$, Forgetful")
plt.semilogy(np.array(list(eta_9.keys())) * 10,eta_9.values(), color=colors[2], marker='.', markevery=4, label=r"$\eta=9$, Forgetful")
plt.semilogy(np.array(list(hmm_eta_1.keys())) * 10,hmm_eta_1.values(), color=colors[0], linestyle='--', marker='x', markevery=4, label=r"$\eta=1$, HMM")
plt.semilogy(np.array(list(hmm_eta_5.keys())) * 10,hmm_eta_5.values(), color=colors[1], linestyle='--', marker='x', markevery=4, label=r"$\eta=5$, HMM")
plt.semilogy(np.array(list(hmm_eta_9.keys())) * 10, hmm_eta_9.values(), color=colors[2], linestyle='--', marker='x', markevery=4, label=r"$\eta=9$, HMM")

plt.grid(which='both', color='grey', linestyle=':', linewidth=0.7)
plt.xlabel(r"Coverage radius ($R$)")
plt.ylabel(r"Average estimation entropy")
plt.legend(ncols=2)
plt.subplots_adjust(left=0.15)
plt.xlim(left=1)
plt.show()

# Load data from the pickle files
with open("results/binary_simulations_eta.pkl", 'rb') as f:
    data_eta = pickle.load(f)

with open("results/binary_simulations_eta_special.pkl", 'rb') as f:
    data_eta_special = pickle.load(f)

# Define the x values
xs = np.arange(1, 50, 4)
xs_first_part = list(range(1, 10))  # Start range for the first part of x's
new_xs = xs_first_part + list(xs[3:])  # Extend xs with the values starting from index 3

# Extract the relevant data for forgetful and HMM
forgetful_special = data_eta_special["forgetful"]
forgetful_optimal_R_special = [np.argmin(list(forgetful_special[key].values())) for key in forgetful_special.keys()]
forgetful_optimal_H_special = [np.min(list(forgetful_special[key].values())) for key in forgetful_special.keys()]

hmm_special = data_eta_special["hmm"]
hmm_optimal_R_special = [np.argmin(list(hmm_special[key].values())) for key in hmm_special.keys()]
hmm_optimal_H_special = [np.min(list(hmm_special[key].values())) for key in hmm_special.keys()]

forgetful = data_eta["forgetful"]
forgetful_optimal_R = [np.argmin(list(forgetful[key].values())) for key in forgetful.keys()]
forgetful_optimal_R_special.extend(forgetful_optimal_R[3:])  # Extending with data from forgetful after index 3
forgetful_optimal_R = forgetful_optimal_R_special.copy()

forgetful_optimal_H = [np.min(list(forgetful[key].values())) for key in forgetful.keys()]
forgetful_optimal_H_special.extend(forgetful_optimal_H[3:])  # Extending with data from forgetful after index 3
forgetful_optimal_H = forgetful_optimal_H_special.copy()

hmm = data_eta["hmm"]
hmm_optimal_R = [np.argmin(list(hmm[key].values())) for key in hmm.keys()]
hmm_optimal_R_special.extend(hmm_optimal_R[3:])  # Extending with data from hmm after index 3
hmm_optimal_R = hmm_optimal_R_special.copy()

hmm_optimal_H = [np.min(list(hmm[key].values())) for key in hmm.keys()]
hmm_optimal_H_special.extend(hmm_optimal_H[3:])  # Extending with data from hmm after index 3
hmm_optimal_H = hmm_optimal_H_special.copy()

# Plot the data
plt.figure(1)
plt.plot(new_xs, np.array(forgetful_optimal_R)*10, marker='.', label="Forgetful")
plt.plot(new_xs, np.array(hmm_optimal_R)*10, marker='.', label="HMM")
plt.xlabel(r"Asymmetry coefficient ($\eta$)")
plt.ylabel(r"Optimal coverage radius ($R^*$)")
plt.xlim(left=1)
plt.legend()
plt.grid(which='both', color='grey', linestyle=':', linewidth=0.7)
plt.show()

# plot optimal entropy
plt.figure(2)
plt.plot(new_xs, forgetful_optimal_H, marker='.', label="Forgetful")
plt.plot(new_xs, hmm_optimal_H, marker='.', label="HMM")
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

forgetful = data["forgetful"]
hmm = data["hmm"]

xs = np.arange(1,50) *10
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