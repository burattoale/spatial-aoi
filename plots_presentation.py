import pickle
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
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
R_unit = 5
with open("results/binary_source.pkl", 'rb') as f:
    data_eta = pickle.load(f)

# --- Figure 0 ---
plt.figure(0)
eta_1 = data_eta["forgetful"]["eta:1"]
eta_5 = data_eta["forgetful"]["eta:5"]
eta_9 = data_eta["forgetful"]["eta:9"]
hmm_eta_1 = data_eta["hmm"]["eta:1"]
hmm_eta_5 = data_eta["hmm"]["eta:5"]
hmm_eta_9 = data_eta["hmm"]["eta:9"]

plt.semilogy(np.array(list(eta_1.keys())) * R_unit,list(eta_1.values()), color=colors[0], marker='.', markevery=4, label=r"$\eta=1$, Forgetful")
plt.semilogy(np.array(list(eta_5.keys())) * R_unit,list(eta_5.values()), color=colors[1], marker='.', markevery=4, label=r"$\eta=5$, Forgetful")
plt.semilogy(np.array(list(eta_9.keys())) * R_unit,list(eta_9.values()), color=colors[2], marker='.', markevery=4, label=r"$\eta=9$, Forgetful")
plt.semilogy(np.array(list(hmm_eta_1.keys())) * R_unit,list(hmm_eta_1.values()), color=colors[0], linestyle='--', marker='x', markevery=4, label=r"$\eta=1$, HMM")
plt.semilogy(np.array(list(hmm_eta_5.keys())) * R_unit,list(hmm_eta_5.values()), color=colors[1], linestyle='--', marker='x', markevery=4, label=r"$\eta=5$, HMM")
plt.semilogy(np.array(list(hmm_eta_9.keys())) * R_unit, list(hmm_eta_9.values()), color=colors[2], linestyle='--', marker='x', markevery=4, label=r"$\eta=9$, HMM")

plt.grid(which='both', color='grey', linestyle=':', linewidth=0.7)
plt.xlabel(r"Coverage radius ($R$)")
plt.ylabel(r"Average estimation entropy")
plt.legend(ncols=2)
plt.subplots_adjust(left=0.15)
plt.xlim(left=1)
plt.show()

# Tikz generation for Figure 0
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
source_counts = ["binary", "three", "five", "seven", "ten"]
plots_fig100 = []
for idx, n_sources in enumerate(source_counts):
    with open(f"results/{n_sources}_source.pkl", 'rb') as f:
        data_multi = pickle.load(f)
    hmm_eta_1_multi = data_multi["hmm"]["eta:1"]
    x_vals = np.array(list(hmm_eta_1_multi.keys())) * R_unit
    y_vals = list(hmm_eta_1_multi.values())
    plt.semilogy(x_vals, y_vals, color=colors[idx], marker='x', markevery=4, label=f"{n_sources} sources, HMM")
    save_to_file(x_vals, y_vals, f"fig100_hmm_{n_sources}.txt", f"{n_sources} sources, HMM")
    plots_fig100.append({'data_file': f'fig100_hmm_{n_sources}.txt', 'legend_entry': f'{n_sources} sources, HMM'})

    try:
        with open(f"results/{n_sources}_source_loc_aware.pkl", 'rb') as f_loc:
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
    plt.semilogy(x_alpha, y_alpha, color=colors_alpha[idx], marker='.', markevery=4, label=fr"${alpha_key}$, HMM")
    save_to_file(x_alpha, y_alpha, f"fig4_hmm_{alpha_key.replace(':', '')}.txt", fr"${alpha_key}$, HMM")
    plots_fig4.append({'data_file': f"fig4_hmm_{alpha_key.replace(':', '')}.txt", 'legend_entry': fr'${alpha_key}$, HMM'})

    x_loc_alpha = np.array(list(eta_loc_alpha.keys())) * R_unit
    y_loc_alpha = list(eta_loc_alpha.values())
    plt.semilogy(x_loc_alpha, y_loc_alpha, color=colors_alpha[idx], linestyle='--', marker='x', markevery=4, label=fr"${alpha_key}$, HMM (Loc-Aware)")
    save_to_file(x_loc_alpha, y_loc_alpha, f"fig4_hmm_{alpha_key.replace(':', '')}_loc_aware.txt", fr"${alpha_key}$, HMM (Loc-Aware)")
    plots_fig4.append({'data_file': f"fig4_hmm_{alpha_key.replace(':', '')}_loc_aware.txt", 'legend_entry': fr'${alpha_key}$, HMM (Loc-Aware)', 'style': 'dashed'})

plt.grid(which='both', color='grey', linestyle=':', linewidth=0.7)
plt.xlabel(r"Coverage radius ($R$)")
plt.ylabel(r"Average estimation entropy")
plt.legend(ncols=2)
plt.subplots_adjust(left=0.15)
plt.xlim(left=1)
plt.show()
generate_tex_file("fig4_H_vs_R_alphas", r"Coverage radius ($R$)", r"Average estimation entropy", plots_fig4, ymode='log', legend_at='(1,0)', legend_anchor='south east', xmin=0)


# --- Figure 10 ---
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

plots_fig10 = [
    {'data_file': 'fig10_err_eta1.txt', 'legend_entry': r'$\eta=1$, HMM'},
    {'data_file': 'fig10_err_eta5.txt', 'legend_entry': r'$\eta=5$, HMM'},
    {'data_file': 'fig10_err_eta9.txt', 'legend_entry': r'$\eta=9$, HMM'},
    {'data_file': 'fig10_err_eta25.txt', 'legend_entry': r'$\eta=25$, HMM'},
]
save_to_file(np.array(list(err_eta_1.keys())) * R_unit, list(err_eta_1.values()), "fig10_err_eta1.txt", r"$\eta=1$, HMM")
save_to_file(np.array(list(err_eta_5.keys())) * R_unit, list(err_eta_5.values()), "fig10_err_eta5.txt", r"$\eta=5$, HMM")
save_to_file(np.array(list(err_eta_9.keys())) * R_unit, list(err_eta_9.values()), "fig10_err_eta9.txt", r"$\eta=9$, HMM")
save_to_file(np.array(list(err_eta_25.keys())) * R_unit, list(err_eta_25.values()), "fig10_err_eta25.txt", r"$\eta=25$, HMM")
generate_tex_file("fig10_Error_vs_R_etas", r"Coverage radius ($R$)", r"Average estimation error", plots_fig10, ymode='log', legend_at='(1,0)', legend_anchor='south east', xmin=0)


# --- Figure 1 & 2 ---
xs = [1,5,9,25]
forgetful = data_eta["forgetful"]
forgetful_optimal_R = [np.argmin(list(forgetful[key].values())) for key in forgetful.keys()]
forgetful_optimal_H = [np.min(list(forgetful[key].values())) for key in forgetful.keys()]
hmm = data_eta["hmm"]
hmm_optimal_R = [np.argmin(list(hmm[key].values())) for key in hmm.keys()]
hmm_optimal_H = [np.min(list(hmm[key].values())) for key in hmm.keys()]

# Plot for Figure 1
plt.figure(1)
plt.plot(xs, np.array(forgetful_optimal_R)* R_unit, marker='.', label="Forgetful")
plt.plot(xs, np.array(hmm_optimal_R)* R_unit, marker='.', label="HMM")
plt.xlabel(r"Asymmetry coefficient ($\eta$)")
plt.ylabel(r"Optimal coverage radius ($R^*$)")
plt.xlim(left=1)
plt.legend()
plt.grid(which='both', color='grey', linestyle=':', linewidth=0.7)
plt.show()

plots_fig1 = [
    {'data_file': 'fig1_forgetful_optimal_R.txt', 'legend_entry': 'Forgetful'},
    {'data_file': 'fig1_hmm_optimal_R.txt', 'legend_entry': 'HMM'},
]
save_to_file(xs, np.array(forgetful_optimal_R) * R_unit, "fig1_forgetful_optimal_R.txt", "Forgetful")
save_to_file(xs, np.array(hmm_optimal_R) * R_unit, "fig1_hmm_optimal_R.txt", "HMM")
generate_tex_file("fig1_Optimal_R_vs_Eta", r"Asymmetry coefficient ($\eta$)", r"Optimal coverage radius ($R^*$)", plots_fig1, legend_at='(1,0)', legend_anchor='south east', xmin=0)

# Plot for Figure 2
plt.figure(2)
plt.plot(xs, forgetful_optimal_H, marker='.', label="Forgetful")
plt.plot(xs, hmm_optimal_H, marker='.', label="HMM")
plt.xlabel(r"Asymmetry coefficient ($\eta$)")
plt.ylabel(r"Minimum entropy ($H^*$)")
plt.grid(which='both', color='grey', linestyle=':', linewidth=0.7)
plt.legend()
plt.xlim(left=1)
plt.show()

plots_fig2 = [
    {'data_file': 'fig2_forgetful_optimal_H.txt', 'legend_entry': 'Forgetful'},
    {'data_file': 'fig2_hmm_optimal_H.txt', 'legend_entry': 'HMM'},
]
save_to_file(xs, forgetful_optimal_H, "fig2_forgetful_optimal_H.txt", "Forgetful")
save_to_file(xs, hmm_optimal_H, "fig2_hmm_optimal_H.txt", "HMM")
generate_tex_file("fig2_Optimal_H_vs_Eta", r"Asymmetry coefficient ($\eta$)", r"Minimum entropy ($H^*$)", plots_fig2, legend_at='(1,0)', legend_anchor='south east', xmin=0)


# --- Figure 3 ---
plt.figure(3)
with open("results/binary_source.pkl", 'rb') as f:
    data = pickle.load(f)
R_unit_fig3 = 10
forgetful = data["forgetful"]
hmm = data["hmm"]

xs_fig3 = np.arange(10,50) * R_unit_fig3
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
generate_tex_file("fig0_H_vs_R_etas", r"Coverage radius ($R$)", r"Average estimation entropy", plots_fig0, ymode='log', legend_pos='south east', xmin=0)

# Plot for eta:1 with different number of sources
plt.figure(100)
source_counts = ["binary", "three", "five", "seven", "ten"]  # Example source counts, adjust as needed
plots_fig100 = []
for idx, n_sources in enumerate(source_counts):
    with open(f"results/{n_sources}_source.pkl", 'rb') as f:
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
        with open(f"results/{n_sources}_source_loc_aware.pkl", 'rb') as f_loc:
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