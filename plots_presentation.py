import pickle
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from plotting_utils import save_to_file, generate_tex_file
import os

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
with open("results/binary_spawc.pkl", 'rb') as f:
    data_eta = pickle.load(f)

# --- Figure 0 ---
plt.figure(0)
with open("results/binary_source_alpha.pkl", 'rb') as f_alpha:
    data_eta_alpha = pickle.load(f_alpha)
hmm_alpha_01 = data_eta_alpha["hmm"]["alpha:0.1"]

plt.semilogy(np.array(list(hmm_alpha_01.keys())) * R_unit, list(hmm_alpha_01.values()), color=colors[3], linestyle='--', marker='x', markevery=4, label=r"$\alpha=0.1$, HMM (binary_alpha)")
hmm_alpha_002 = data_eta_alpha["hmm"]["alpha:0.02"]
plt.semilogy(np.array(list(hmm_alpha_002.keys())) * R_unit, list(hmm_alpha_002.values()), color=colors[4], linestyle='--', marker='x', markevery=4, label=r"$\alpha=0.02$, HMM (binary_alpha)")
eta_5 = data_eta["forgetful"]["eta:5"]
eta_9 = data_eta["forgetful"]["eta:20"]
hmm_eta_1 = data_eta["hmm"]["eta:1"]
hmm_eta_5 = data_eta["hmm"]["eta:5"]
hmm_eta_9 = data_eta["hmm"]["eta:20"]

#plt.semilogy(np.array(list(eta_1.keys())) * R_unit,list(eta_1.values()), color=colors[0], marker='.', markevery=4, label=r"$\eta=1$, Forgetful")
#plt.semilogy(np.array(list(eta_5.keys())) * R_unit,list(eta_5.values()), color=colors[1], marker='.', markevery=4, label=r"$\eta=5$, Forgetful")
#plt.semilogy(np.array(list(eta_9.keys())) * R_unit,list(eta_9.values()), color=colors[2], marker='.', markevery=4, label=r"$\eta=9$, Forgetful")
#plt.semilogy(np.array(list(hmm_eta_1.keys())) * R_unit,list(hmm_eta_1.values()), color=colors[0], linestyle='--', marker='x', markevery=4, label=r"$\eta=1$, HMM")
#plt.semilogy(np.array(list(hmm_eta_5.keys())) * R_unit,list(hmm_eta_5.values()), color=colors[1], linestyle='--', marker='x', markevery=4, label=r"$\eta=5$, HMM")
#plt.semilogy(np.array(list(hmm_eta_9.keys())) * R_unit, list(hmm_eta_9.values()), color=colors[2], linestyle='--', marker='x', markevery=4, label=r"$\eta=9$, HMM")

plt.grid(which='both', color='grey', linestyle=':', linewidth=0.7)
plt.xlabel(r"Coverage radius ($R$)")
plt.ylabel(r"Average estimation entropy")
plt.legend(ncols=2)
plt.subplots_adjust(left=0.15)
plt.xlim(left=1)
plt.savefig("final_plots/forgetful_vs_hmm.png")
plt.show()

# Create subfolder for txt traces if it doesn't exist
tikz_folder = "final_plots/fig0_traces"
os.makedirs(tikz_folder, exist_ok=True)

# Save only the two HMM alpha plots as txt traces for TikZ
save_to_file(np.array(list(hmm_alpha_01.keys())) * R_unit, list(hmm_alpha_01.values()), os.path.join(tikz_folder, "hmm_alpha_01.txt"), r"$\alpha=0.1$, HMM (binary_alpha)")
save_to_file(np.array(list(hmm_alpha_002.keys())) * R_unit, list(hmm_alpha_002.values()), os.path.join(tikz_folder, "hmm_alpha_002.txt"), r"$\alpha=0.02$, HMM (binary_alpha)")
plots_fig0 = [
    {'data_file': 'fig0_forgetful_eta1.txt', 'legend_entry': r'$\eta=1$, Forgetful', 'color': 'blue', 'style': 'solid'},
    {'data_file': 'fig0_forgetful_eta5.txt', 'legend_entry': r'$\eta=5$, Forgetful', 'color': 'orange', 'style': 'solid'},
    {'data_file': 'fig0_forgetful_eta9.txt', 'legend_entry': r'$\eta=9$, Forgetful', 'color': 'green', 'style': 'solid'},
    {'data_file': 'fig0_hmm_eta1.txt', 'legend_entry': r'$\eta=1$, HMM', 'color': 'blue', 'style': 'dashed'},
    {'data_file': 'fig0_hmm_eta5.txt', 'legend_entry': r'$\eta=5$, HMM', 'color': 'orange', 'style': 'dashed'},
    {'data_file': 'fig0_hmm_eta9.txt', 'legend_entry': r'$\eta=9$, HMM', 'color': 'green', 'style': 'dashed'},
]
save_to_file(np.array(list(eta_1.keys())) * R_unit, list(eta_1.values()), "fig0_forgetful_eta1.txt", r"$\eta=1$, Forgetful")
save_to_file(np.array(list(eta_5.keys())) * R_unit, list(eta_5.values()), "fig0_forgetful_eta5.txt", r"$\eta=5$, Forgetful")
save_to_file(np.array(list(eta_9.keys())) * R_unit, list(eta_9.values()), "fig0_forgetful_eta9.txt", r"$\eta=9$, Forgetful")
save_to_file(np.array(list(hmm_eta_1.keys())) * R_unit, list(hmm_eta_1.values()), "fig0_hmm_eta1.txt", r"$\eta=1$, HMM")
save_to_file(np.array(list(hmm_eta_5.keys())) * R_unit, list(hmm_eta_5.values()), "fig0_hmm_eta5.txt", r"$\eta=5$, HMM")
save_to_file(np.array(list(hmm_eta_9.keys())) * R_unit, list(hmm_eta_9.values()), "fig0_hmm_eta9.txt", r"$\eta=9$, HMM")
generate_tex_file("fig0_H_vs_R_etas", r"Coverage radius ($R$)", r"Average estimation entropy", plots_fig0, ymode='log', legend_at='(1,0)', legend_anchor='south east', xmin=0)


# --- Figure 100 ---
plt.figure(100)
source_counts = ["binary_spawc", "four_spawc", "eight_spawc"]
plots_fig100 = []
for idx, n_sources in enumerate(source_counts):
    with open(f"results/{n_sources}.pkl", 'rb') as f:
        data_multi = pickle.load(f)
    hmm_eta_1_multi = data_multi["hmm"]["eta:1"]
    x_vals = np.array(list(hmm_eta_1_multi.keys())) * R_unit
    y_vals = list(hmm_eta_1_multi.values())
    plt.semilogy(x_vals, y_vals, color=colors[idx], marker='x', markevery=4, label=f"{n_sources} sources, HMM")
    save_to_file(x_vals, y_vals, f"fig100_hmm_{n_sources}.txt", f"{n_sources} sources, HMM")
    plots_fig100.append({'data_file': f'fig100_hmm_{n_sources}.txt', 'legend_entry': f'{n_sources} sources, HMM'})

    try:
        with open(f"results/{n_sources}_loc_aware.pkl", 'rb') as f_loc:
            data_multi_loc = pickle.load(f_loc)
        hmm_eta_1_multi_loc = data_multi_loc["hmm"]["eta:1"]
        x_vals_loc = np.array(list(hmm_eta_1_multi_loc.keys())) * R_unit
        y_vals_loc = list(hmm_eta_1_multi_loc.values())
        plt.semilogy(x_vals_loc, y_vals_loc, color=colors[idx], linestyle='--', marker='o', markevery=4, label=f"{n_sources} sources, HMM (Loc-Aware)")
        save_to_file(x_vals_loc, y_vals_loc, f"fig100_hmm_{n_sources}_loc_aware.txt", f"{n_sources} sources, HMM (Loc-Aware)")
        plots_fig100.append({'data_file': f'fig100_hmm_{n_sources}_loc_aware.txt', 'legend_entry': f'{n_sources} sources, HMM (Loc-Aware)', 'style': 'dashed'})
    except FileNotFoundError:
        pass
plt.grid(which='both', color='grey', linestyle=':', linewidth=0.7)
plt.xlabel(r"Coverage radius ($R$)")
plt.ylabel(r"Average estimation entropy")
plt.legend()
plt.xlim(left=1)
plt.savefig("final_plots/source_cardinality.png")
plt.show()
generate_tex_file("fig100_H_vs_R_sources", r"Coverage radius ($R$)", r"Average estimation entropy", plots_fig100, ymode='log', legend_at='(1,0)', legend_anchor='south east', xmin=0)


# --- Figure 4 ---
plt.figure(4)
with open("results/binary_source_alpha.pkl", 'rb') as f_alpha:
    data_eta_alpha = pickle.load(f_alpha)
with open("results/binary_source_loc_aware_alpha.pkl", 'rb') as f_alpha_loc:
    data_eta_loc_alpha = pickle.load(f_alpha_loc)

alpha_keys = ["alpha:0.02", "alpha:0.05", "alpha:0.1"]
colors_alpha = colors[:len(alpha_keys)]
plots_fig4 = []

for idx, alpha_key in enumerate(alpha_keys):
    eta_alpha = data_eta_alpha["hmm"][alpha_key]
    eta_loc_alpha = data_eta_loc_alpha["hmm"][alpha_key]
    
    x_alpha = np.array(list(eta_alpha.keys())) * R_unit
    y_alpha = list(eta_alpha.values())
    plt.plot(x_alpha, y_alpha, color=colors_alpha[idx], marker='.', markevery=4, label=fr"${alpha_key}$, HMM")
    save_to_file(x_alpha, y_alpha, f"fig4_hmm_{alpha_key.replace(':', '')}.txt", fr"${alpha_key}$, HMM")
    plots_fig4.append({'data_file': f"fig4_hmm_{alpha_key.replace(':', '')}.txt", 'legend_entry': fr'${alpha_key}$, HMM'})

    x_loc_alpha = np.array(list(eta_loc_alpha.keys())) * R_unit
    y_loc_alpha = list(eta_loc_alpha.values())
    plt.plot(x_loc_alpha, y_loc_alpha, color=colors_alpha[idx], linestyle='--', marker='x', markevery=4, label=fr"${alpha_key}$, HMM (Loc-Aware)")
    save_to_file(x_loc_alpha, y_loc_alpha, f"fig4_hmm_{alpha_key.replace(':', '')}_loc_aware.txt", fr"${alpha_key}$, HMM (Loc-Aware)")
    plots_fig4.append({'data_file': f"fig4_hmm_{alpha_key.replace(':', '')}_loc_aware.txt", 'legend_entry': fr'${alpha_key}$, HMM (Loc-Aware)', 'style': 'dashed'})

plt.grid(which='both', color='grey', linestyle=':', linewidth=0.7)
plt.xlabel(r"Coverage radius ($R$)")
plt.ylabel(r"Average estimation entropy")
plt.legend(ncols=2)
plt.subplots_adjust(left=0.15)
plt.xlim(left=1)
plt.savefig("final_plots/alpha_loc_aware.png")
plt.show()
generate_tex_file("fig4_H_vs_R_alphas", r"Coverage radius ($R$)", r"Average estimation entropy", plots_fig4, ymode='log', legend_at='(1,0)', legend_anchor='south east', xmin=0)


# --- Figure 10 ---
plt.figure(10)
err_eta_1 = data_eta["hmm_err"]["eta:1"]
err_eta_5 = data_eta["hmm_err"]["eta:5"]
err_eta_9 = data_eta["hmm_err"]["eta:20"]
#err_eta_25 = data_eta["hmm_err"]["eta:25"]

plt.semilogy(np.array(list(err_eta_1.keys())) * R_unit, err_eta_1.values(), color=colors[0], marker='.', markevery=4, label=r"$\eta=1$, HMM")
plt.semilogy(np.array(list(err_eta_5.keys())) * R_unit, err_eta_5.values(), color=colors[1], marker='d', markevery=4, label=r"$\eta=5$, HMM")
plt.semilogy(np.array(list(err_eta_9.keys())) * R_unit, err_eta_9.values(), color=colors[2], marker='*', markevery=4, label=r"$\eta=9$, HMM")
#plt.semilogy(np.array(list(err_eta_25.keys())) * R_unit, err_eta_25.values(), color=colors[3], marker='x', markevery=4, label=r"$\eta=25$, HMM")
plt.grid(which='both', color='grey', linestyle=':', linewidth=0.7)
plt.legend()
plt.xlabel(r"Coverage radius ($R$)")
plt.ylabel(r"Average estimation error")
plt.savefig("final_plots/estimation_error.png")
plt.show()

plots_fig10 = [
    {'data_file': 'fig10_err_eta1.txt', 'legend_entry': r'$\eta=1$, HMM'},
    {'data_file': 'fig10_err_eta5.txt', 'legend_entry': r'$\eta=5$, HMM'},
    {'data_file': 'fig10_err_eta9.txt', 'legend_entry': r'$\eta=9$, HMM'},
    {'data_file': 'fig10_err_eta25.txt', 'legend_entry': r'$\eta=25$, HMM'},
]
save_to_file(np.array(list(err_eta_1.keys())) * R_unit, list(err_eta_1.values()), "fig10_err_eta1.txt", r"$\eta=1$, HMM")
save_to_file(np.array(list(err_eta_5.keys())) * R_unit, list(err_eta_5.values()), "fig10_err_eta5.txt", r"$\eta=5$, HMM")
save_to_file(np.array(list(err_eta_9.keys())) * R_unit, list(err_eta_9.values()), "fig10_err_eta9.txt", r"$\eta=9$, HMM")
#save_to_file(np.array(list(err_eta_25.keys())) * R_unit, list(err_eta_25.values()), "fig10_err_eta25.txt", r"$\eta=25$, HMM")
generate_tex_file("fig10_Error_vs_R_etas", r"Coverage radius ($R$)", r"Average estimation error", plots_fig10, ymode='log', legend_at='(1,0)', legend_anchor='south east', xmin=0)


# --- Figure 1 & 2 ---
# --- Figure 1 & 2 ---
with open("results/binary_spawc.pkl", 'rb') as f_all_eta:
    data_eta_all = pickle.load(f_all_eta)

# Extract eta values from keys
forgetful = data_eta_all["forgetful"]
hmm = data_eta_all["hmm"]

eta_keys = sorted([int(k.split(":")[1]) for k in forgetful.keys()])
forgetful_optimal_R = []
forgetful_optimal_H = []
hmm_optimal_R = []
hmm_optimal_H = []

for eta in eta_keys:
    key = f"eta:{eta}"
    # Forgetful
    vals_forget = list(forgetful[key].values())
    keys_forget = list(forgetful[key].keys())
    min_idx_forget = np.argmin(vals_forget)
    forgetful_optimal_R.append(keys_forget[min_idx_forget])
    forgetful_optimal_H.append(vals_forget[min_idx_forget])
    # HMM
    vals_hmm = list(hmm[key].values())
    keys_hmm = list(hmm[key].keys())
    min_idx_hmm = np.argmin(vals_hmm)
    hmm_optimal_R.append(keys_hmm[min_idx_hmm])
    hmm_optimal_H.append(vals_hmm[min_idx_hmm])

# Plot for Figure 1
plt.figure(1)
plt.plot(eta_keys, np.array(forgetful_optimal_R) * R_unit, marker='.', label="Forgetful")
plt.plot(eta_keys, np.array(hmm_optimal_R) * R_unit, marker='.', label="HMM")
plt.xlabel(r"Asymmetry coefficient ($\eta$)")
plt.ylabel(r"Optimal coverage radius ($R^*$)")
plt.xlim(left=min(eta_keys))
plt.legend()
plt.grid(which='both', color='grey', linestyle=':', linewidth=0.7)
plt.savefig("final_plots/hmm_optimal_R.png")
plt.show()

plots_fig1 = [
    {'data_file': 'fig1_forgetful_optimal_R.txt', 'legend_entry': 'Forgetful'},
    {'data_file': 'fig1_hmm_optimal_R.txt', 'legend_entry': 'HMM'},
]
save_to_file(eta_keys, np.array(forgetful_optimal_R) * R_unit, "fig1_forgetful_optimal_R.txt", "Forgetful")
save_to_file(eta_keys, np.array(hmm_optimal_R) * R_unit, "fig1_hmm_optimal_R.txt", "HMM")
generate_tex_file("fig1_Optimal_R_vs_Eta", r"Asymmetry coefficient ($\eta$)", r"Optimal coverage radius ($R^*$)", plots_fig1, legend_at='(1,0)', legend_anchor='south east', xmin=0)

# Plot for Figure 2
plt.figure(2)
plt.plot(eta_keys, forgetful_optimal_H, marker='.', label="Forgetful")
plt.plot(eta_keys, hmm_optimal_H, marker='.', label="HMM")
plt.xlabel(r"Asymmetry coefficient ($\eta$)")
plt.ylabel(r"Minimum entropy ($H^*$)")
plt.grid(which='both', color='grey', linestyle=':', linewidth=0.7)
plt.legend()
plt.xlim(left=min(eta_keys))
plt.savefig("final_plots/hmm_optimal_H.png")
plt.show()

plots_fig2 = [
    {'data_file': 'fig2_forgetful_optimal_H.txt', 'legend_entry': 'Forgetful'},
    {'data_file': 'fig2_hmm_optimal_H.txt', 'legend_entry': 'HMM'},
]
save_to_file(eta_keys, forgetful_optimal_H, "fig2_forgetful_optimal_H.txt", "Forgetful")
save_to_file(eta_keys, hmm_optimal_H, "fig2_hmm_optimal_H.txt", "HMM")
generate_tex_file("fig2_Optimal_H_vs_Eta", r"Asymmetry coefficient ($\eta$)", r"Minimum entropy ($H^*$)", plots_fig2, legend_at='(1,0)', legend_anchor='south east', xmin=0)


# --- Figure 3 ---
plt.figure(3)
with open("results/binary_spawc.pkl", 'rb') as f:
    data = pickle.load(f)
R_unit_fig3 = 10
forgetful = data["forgetful"]
hmm = data["hmm"]

xs_fig3 = np.arange(1,1+len(list(hmm["eta:1"].values()))) * R_unit_fig3
values_forget = np.array(list(forgetful["eta:1"].values()))
values_hmm = np.array(list(hmm["eta:1"].values()))
min_forget = np.min(values_forget)
idx_forget = np.argmin(values_forget)
min_hmm = np.min(values_hmm)
idx_hmm = np.argmin(values_hmm)
plt.semilogy(xs_fig3, values_forget, color=colors[0], marker='.', markevery=4, label="Forgetful")
plt.semilogy(xs_fig3, values_hmm, color=colors[1], marker='.', markevery=4, label="HMM")
plt.semilogy(xs_fig3[idx_forget:], [min_forget]*len(xs_fig3[idx_forget:]), color=colors[0], linestyle=':', label=r"Forgetful, optimal $\zeta^*$")
plt.semilogy(xs_fig3[idx_hmm:], [min_hmm]*len(xs_fig3[idx_hmm:]), color=colors[1], linestyle=':', label=r"HMM, optimal $\zeta^*$")
plt.grid(which='both', color='grey', linestyle=':', linewidth=0.7)
plt.xlabel(r"Coverage radius ($R$)")
plt.ylabel(r"Average estimation entropy")
plt.legend(ncols=2)
plt.xlim(left=1)
plt.savefig("final_plots/tx_prob_opt.png")
plt.show()


plots_fig3 = [
    {'data_file': 'fig3_forgetful.txt', 'legend_entry': 'Forgetful'},
    {'data_file': 'fig3_hmm.txt', 'legend_entry': 'HMM'},
    {'data_file': 'fig3_forgetful_optimal.txt', 'legend_entry': r'Forgetful, optimal $\zeta^*$', 'style': 'dashed'},
    {'data_file': 'fig3_hmm_optimal.txt', 'legend_entry': r'HMM, optimal $\zeta^*$', 'style': 'dashed'},
]
save_to_file(xs_fig3, values_forget, "fig3_forgetful.txt", "Forgetful")
save_to_file(xs_fig3, values_hmm, "fig3_hmm.txt", "HMM")
save_to_file(xs_fig3[idx_forget:], [min_forget]*len(xs_fig3[idx_forget:]), "fig3_forgetful_optimal.txt", r"Forgetful, optimal $\zeta^*$")
save_to_file(xs_fig3[idx_hmm:], [min_hmm]*len(xs_fig3[idx_hmm:]), "fig3_hmm_optimal.txt", r"HMM, optimal $\zeta^*$")
generate_tex_file("fig3_H_vs_R_optimal_zeta", r"Coverage radius ($R$)", r"Average estimation entropy", plots_fig3, ymode='log', legend_at='(1,0)', legend_anchor='south east', xmin=0)


# Tikz generation
plots_fig0 = [
    {'data_file': 'fig0_forgetful_eta1.txt', 'legend_entry': r'$\eta=1$, Forgetful', 'color': 'blue', 'style': 'solid'},
    {'data_file': 'fig0_forgetful_eta5.txt', 'legend_entry': r'$\eta=5$, Forgetful', 'color': 'orange', 'style': 'solid'},
    {'data_file': 'fig0_forgetful_eta9.txt', 'legend_entry': r'$\eta=9$, Forgetful', 'color': 'green', 'style': 'solid'},
    {'data_file': 'fig0_hmm_eta1.txt', 'legend_entry': r'$\eta=1$, HMM', 'color': 'blue', 'style': 'dashed'},
    {'data_file': 'fig0_hmm_eta5.txt', 'legend_entry': r'$\eta=5$, HMM', 'color': 'orange', 'style': 'dashed'},
    {'data_file': 'fig0_hmm_eta9.txt', 'legend_entry': r'$\eta=9$, HMM', 'color': 'green', 'style': 'dashed'},
]
save_to_file(np.array(list(eta_1.keys())) * R_unit, list(eta_1.values()), "fig0_forgetful_eta1.txt", r"$\eta=1$, Forgetful")
save_to_file(np.array(list(eta_5.keys())) * R_unit, list(eta_5.values()), "fig0_forgetful_eta5.txt", r"$\eta=5$, Forgetful")
save_to_file(np.array(list(eta_9.keys())) * R_unit, list(eta_9.values()), "fig0_forgetful_eta9.txt", r"$\eta=9$, Forgetful")
save_to_file(np.array(list(hmm_eta_1.keys())) * R_unit, list(hmm_eta_1.values()), "fig0_hmm_eta1.txt", r"$\eta=1$, HMM")
save_to_file(np.array(list(hmm_eta_5.keys())) * R_unit, list(hmm_eta_5.values()), "fig0_hmm_eta5.txt", r"$\eta=5$, HMM")
save_to_file(np.array(list(hmm_eta_9.keys())) * R_unit, list(hmm_eta_9.values()), "fig0_hmm_eta9.txt", r"$\eta=9$, HMM")
generate_tex_file("fig0_H_vs_R_etas", r"Coverage radius ($R$)", r"Average estimation entropy", plots_fig0, ymode='log', xmin=0)

# Plot for eta:1 with different number of sources
plt.figure(100)
source_counts = ["binary_spawc", "four_spawc", "eight_spawc"]  # Example source counts, adjust as needed
plots_fig100 = []
for idx, n_sources in enumerate(source_counts):
    with open(f"results/{n_sources}.pkl", 'rb') as f:
        data_multi = pickle.load(f)
    hmm_eta_1_multi = data_multi["hmm"]["eta:1"]
    x_vals = np.array(list(hmm_eta_1_multi.keys())) * R_unit
    y_vals = list(hmm_eta_1_multi.values())
    plt.semilogy(x_vals, y_vals,
                 color=colors[idx], marker='x', markevery=4, label=f"{n_sources} sources, HMM")
    save_to_file(x_vals, y_vals, f"fig100_hmm_{n_sources}.txt", f"{n_sources} sources, HMM")
    plots_fig100.append({'data_file': f'fig100_hmm_{n_sources}.txt', 'legend_entry': f'{n_sources} sources, HMM'})

    # Location-aware comparison
    try:
        with open(f"results/{n_sources}_loc_aware.pkl", 'rb') as f_loc:
            data_multi_loc = pickle.load(f_loc)
        hmm_eta_1_multi_loc = data_multi_loc["hmm"]["eta:1"]
        x_vals_loc = np.array(list(hmm_eta_1_multi_loc.keys())) * R_unit
        y_vals_loc = list(hmm_eta_1_multi_loc.values())
        plt.semilogy(x_vals_loc, y_vals_loc,
                     color=colors[idx], linestyle='--', marker='o', markevery=4, label=f"{n_sources} sources, HMM (Loc-Aware)")
        save_to_file(x_vals_loc, y_vals_loc, f"fig100_hmm_{n_sources}_loc_aware.txt", f"{n_sources} sources, HMM (Loc-Aware)")
        plots_fig100.append({'data_file': f'fig100_hmm_{n_sources}_loc_aware.txt', 'legend_entry': f'{n_sources} sources, HMM (Loc-Aware)', 'style': 'dashed'})
    except FileNotFoundError:
        pass  # Skip if location-aware data is not available
plt.grid(which='both', color='grey', linestyle=':', linewidth=0.7)
plt.xlabel(r"Coverage radius ($R$)")
plt.ylabel(r"Average estimation entropy")
plt.legend()
plt.xlim(left=1)
plt.show()

# --- New Figure: Multiple eta for multi-symbol sources ---
plt.figure(101)
multi_eta_values = ["eta:1", "eta:5", "eta:9"]
multi_source_counts = ["four_spawc", "eight_spawc"]  # Exclude binary for clarity
for idx, n_sources in enumerate(multi_source_counts):
    with open(f"results/{n_sources}.pkl", 'rb') as f:
        data_multi = pickle.load(f)
    for jdx, eta_val in enumerate(multi_eta_values):
        if eta_val in data_multi["hmm"]:
            hmm_multi = data_multi["hmm"][eta_val]
            x_vals = np.array(list(hmm_multi.keys())) * R_unit
            y_vals = list(hmm_multi.values())
            plt.semilogy(x_vals, y_vals,
                         color=colors[idx], linestyle=['-', '--', ':'][jdx], marker=['x', 'o', 's'][jdx], markevery=4,
                         label=f"{n_sources} sources, {eta_val}")
plt.grid(which='both', color='grey', linestyle=':', linewidth=0.7)
plt.xlabel(r"Coverage radius ($R$)")
plt.ylabel(r"Average estimation entropy")
plt.legend()
plt.xlim(left=1)
plt.title("Multi-symbol sources: Multiple $\eta$")
plt.savefig("final_plots/multi_symbol_multi_eta.png")
plt.show()

# --- New Figure: Minimum entropy vs eta for multi-symbol sources ---
plt.figure(102)
multi_source_counts = ["four_spawc", "eight_spawc"]
multi_eta_values = ["eta:1", "eta:5", "eta:9"]
eta_numeric = [1, 5, 9]
for idx, n_sources in enumerate(multi_source_counts):
    min_entropies = []
    for eta_val in multi_eta_values:
        try:
            with open(f"results/{n_sources}.pkl", 'rb') as f:
                data_multi = pickle.load(f)
            if eta_val in data_multi["hmm"]:
                hmm_multi = data_multi["hmm"][eta_val]
                min_entropies.append(np.min(list(hmm_multi.values())))
            else:
                min_entropies.append(np.nan)
        except Exception:
            min_entropies.append(np.nan)
    plt.plot(eta_numeric, min_entropies, marker='.', label=f"{n_sources} sources")
plt.xlabel(r"Asymmetry coefficient ($\eta$)")
plt.ylabel(r"Minimum entropy ($H^*$)")
plt.grid(which='both', color='grey', linestyle=':', linewidth=0.7)
plt.legend()
plt.xlim(left=1)
plt.title("Minimum entropy vs $\eta$ for multi-symbol sources")
plt.savefig("final_plots/min_H_multi_symbol_multi_eta.png")
plt.show()

# --- New Figure: Optimal coverage radius vs eta for multi-symbol sources ---
plt.figure(103)
for idx, n_sources in enumerate(multi_source_counts):
    optimal_radii = []
    for eta_val in multi_eta_values:
        try:
            with open(f"results/{n_sources}.pkl", 'rb') as f:
                data_multi = pickle.load(f)
            if eta_val in data_multi["hmm"]:
                hmm_multi = data_multi["hmm"][eta_val]
                values = list(hmm_multi.values())
                keys = list(hmm_multi.keys())
                min_idx = np.argmin(values)
                optimal_radii.append(keys[min_idx] * R_unit)
            else:
                optimal_radii.append(np.nan)
        except Exception:
            optimal_radii.append(np.nan)
    plt.plot(eta_numeric, optimal_radii, marker='.', label=f"{n_sources} sources")
plt.xlabel(r"Asymmetry coefficient ($\eta$)")
plt.ylabel(r"Optimal coverage radius ($R^*$)")
plt.grid(which='both', color='grey', linestyle=':', linewidth=0.7)
plt.legend()
plt.xlim(left=1)
plt.title("Optimal coverage radius vs $\eta$ for multi-symbol sources")
plt.savefig("final_plots/figure_103.png")
plt.show()