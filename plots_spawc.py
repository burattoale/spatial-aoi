import pickle
import numpy as np
import matplotlib.pyplot as plt
from plotting_utils import save_to_file, generate_tex_file

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
R_unit = 10

try:
    with open("results/binary_spawc.pkl", 'rb') as f:
        data_spawc = pickle.load(f)
except FileNotFoundError:
    print("Error: results/binary_spawc.pkl not found.")
    print("Please run the simulation to generate the data.")
    exit()

# --- Figure 1: Entropy vs Coverage Radius for different etas ---
plt.figure(1)
plots_fig1 = []
schemes = ['hmm', 'forgetful']
for scheme_idx, scheme in enumerate(schemes):
    if scheme in data_spawc:
        eta_keys = sorted(data_spawc[scheme].keys())
        for idx, eta_key in enumerate(eta_keys):
            eta_data = data_spawc[scheme][eta_key]
            x_vals = np.array(list(eta_data.keys())) * R_unit
            y_vals = list(eta_data.values())
            label = f"{scheme}, {eta_key}"
            plt.semilogy(x_vals, y_vals, color=colors[idx], marker='.', linestyle='--' if scheme == 'forgetful' else '-', markevery=4, label=label)
            
            data_filename = f"fig_spawc_{scheme}_{eta_key.replace(':', '')}.txt"
            save_to_file(x_vals, y_vals, data_filename, label)
            plots_fig1.append({'data_file': data_filename, 'legend_entry': label, 'style': 'dashed' if scheme == 'forgetful' else 'solid'})

plt.grid(which='both', color='grey', linestyle=':', linewidth=0.7)
plt.xlabel(r"Coverage radius ($R$)")
plt.ylabel(r"Average estimation entropy")
plt.legend(title="Scheme, Eta")
plt.xlim(left=0)
plt.title("Entropy vs Coverage Radius (SPAMC)")
plt.savefig("final_plots/spawc_entropy_vs_radius.png")
plt.close()

generate_tex_file("fig_spawc_H_vs_R_etas", r"Coverage radius ($R$)", r"Average estimation entropy", plots_fig1, ymode='log', legend_at='(1,0)', legend_anchor='south east', xmin=0)


# --- Figure 2 & 3: Optimal H and R vs Eta ---
optimal_H_hmm = []
optimal_R_hmm = []
optimal_H_forgetful = []
optimal_R_forgetful = []
eta_numeric_hmm = []
eta_numeric_forgetful = []

if 'hmm' in data_spawc:
    eta_keys_hmm = sorted(data_spawc['hmm'].keys(), key=lambda x: int(x.split(':')[1]))
    for eta_key in eta_keys_hmm:
        eta_data = data_spawc['hmm'][eta_key]
        if eta_data:
            radii = list(eta_data.keys())
            entropies = list(eta_data.values())
            min_idx = np.argmin(entropies)
            optimal_H_hmm.append(entropies[min_idx])
            optimal_R_hmm.append(radii[min_idx] * R_unit)
            eta_numeric_hmm.append(int(eta_key.split(':')[1]))

if 'forgetful' in data_spawc:
    eta_keys_forgetful = sorted(data_spawc['forgetful'].keys(), key=lambda x: int(x.split(':')[1]))
    for eta_key in eta_keys_forgetful:
        eta_data = data_spawc['forgetful'][eta_key]
        if eta_data:
            radii = list(eta_data.keys())
            entropies = list(eta_data.values())
            min_idx = np.argmin(entropies)
            optimal_H_forgetful.append(entropies[min_idx])
            optimal_R_forgetful.append(radii[min_idx] * R_unit)
            eta_numeric_forgetful.append(int(eta_key.split(':')[1]))


# Plot for Figure 2: Minimum Entropy vs Eta
plt.figure(2)
if optimal_H_hmm:
    plt.plot(eta_numeric_hmm, optimal_H_hmm, marker='.', label="HMM")
if optimal_H_forgetful:
    plt.plot(eta_numeric_forgetful, optimal_H_forgetful, marker='x', linestyle='--', label="Forgetful")
plt.xlabel(r"Asymmetry coefficient ($\eta$)")
plt.ylabel(r"Minimum entropy ($H^*$)")
plt.grid(which='both', color='grey', linestyle=':', linewidth=0.7)
plt.legend()
if eta_numeric_hmm or eta_numeric_forgetful:
    all_etas = eta_numeric_hmm + eta_numeric_forgetful
    if all_etas:
        plt.xlim(left=min(all_etas))
plt.title("Minimum Entropy vs $\eta$ (SPAMC)")
plt.savefig("final_plots/spawc_min_H_vs_eta.png")
plt.close()

plots_fig2 = []
if optimal_H_hmm:
    plots_fig2.append({'data_file': 'fig_spawc_optimal_H_hmm.txt', 'legend_entry': 'HMM'})
    save_to_file(eta_numeric_hmm, optimal_H_hmm, "fig_spawc_optimal_H_hmm.txt", "HMM")
if optimal_H_forgetful:
    plots_fig2.append({'data_file': 'fig_spawc_optimal_H_forgetful.txt', 'legend_entry': 'Forgetful', 'style': 'dashed'})
    save_to_file(eta_numeric_forgetful, optimal_H_forgetful, "fig_spawc_optimal_H_forgetful.txt", "Forgetful")
if plots_fig2:
    generate_tex_file("fig_spawc_Optimal_H_vs_Eta", r"Asymmetry coefficient ($\eta$)", r"Minimum entropy ($H^*$)", plots_fig2, legend_at='(1,0)', legend_anchor='south east', xmin=0)


# Plot for Figure 3: Optimal Radius vs Eta
plt.figure(3)
if optimal_R_hmm:
    plt.plot(eta_numeric_hmm, optimal_R_hmm, marker='.', label="HMM")
if optimal_R_forgetful:
    plt.plot(eta_numeric_forgetful, optimal_R_forgetful, marker='x', linestyle='--', label="Forgetful")
plt.xlabel(r"Asymmetry coefficient ($\eta$)")
plt.ylabel(r"Optimal coverage radius ($R^*$)")
plt.grid(which='both', color='grey', linestyle=':', linewidth=0.7)
plt.legend()
if eta_numeric_hmm or eta_numeric_forgetful:
    all_etas = eta_numeric_hmm + eta_numeric_forgetful
    if all_etas:
        plt.xlim(left=min(all_etas))
plt.title("Optimal Coverage Radius vs $\eta$ (SPAMC)")
plt.savefig("final_plots/spawc_optimal_R_vs_eta.png")
plt.close()

plots_fig3 = []
if optimal_R_hmm:
    plots_fig3.append({'data_file': 'fig_spawc_optimal_R_hmm.txt', 'legend_entry': 'HMM'})
    save_to_file(eta_numeric_hmm, optimal_R_hmm, "fig_spawc_optimal_R_hmm.txt", "HMM")
if optimal_R_forgetful:
    plots_fig3.append({'data_file': 'fig_spawc_optimal_R_forgetful.txt', 'legend_entry': 'Forgetful', 'style': 'dashed'})
    save_to_file(eta_numeric_forgetful, optimal_R_forgetful, "fig_spawc_optimal_R_forgetful.txt", "Forgetful")
if plots_fig3:
    generate_tex_file("fig_spawc_Optimal_R_vs_Eta", r"Asymmetry coefficient ($\eta$)", r"Optimal coverage radius ($R^*$)", plots_fig3, legend_at='(1,0)', legend_anchor='south east', xmin=0)
