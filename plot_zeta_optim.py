import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from scipy.optimize import least_squares

with open("results/zeta_optim_formulas_partial.pickle", "rb") as f:
    data:dict = pickle.load(f)

optimized_entropy = []
non_optimized_entropy = []
for k in data.keys():
    optimized_entropy.append(data[k]["entropy_opt"])
    non_optimized_entropy.append(data[k]["entropy_same_z"])

plt.figure()
plt.plot(list(data.keys()), optimized_entropy, marker=".", label="Entropy optimized $\zeta_i$")
plt.plot(list(data.keys()), non_optimized_entropy, marker="x", linestyle="--", label="Entropy same $\zeta=5e-4$")
plt.xlabel("Number of regions, K")
plt.ylabel("Estimation entropy")
plt.legend()
plt.grid()
plt.savefig("plots/entropy_opt.png")
plt.show()

folder = "plots/tx_opt"
os.makedirs(folder, exist_ok=True)

for i, k in enumerate(data.keys()):
    plt.figure()
    plt.plot(np.arange(list(data.keys())[i]), data[list(data.keys())[i]]["zeta"].values(), linestyle=":", marker="s", label="optimized $\zeta$ per region")
    plt.xlabel("Index of region")
    plt.ylabel("Transmission probability")
    plt.title(f"Number of regions: {k}")
    plt.legend()
    plt.grid()
    plt.savefig(folder+f"/tx_prob_opt_{i}.png")
    plt.close()

initial_points = []
for val in data.values():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    from sklearn.metrics import r2_score, mean_squared_error
    from scipy.interpolate import UnivariateSpline

    # 1. Prepare the data
    x = np.arange(len(val["zeta"].values()))
    y = np.array(list(val["zeta"].values()))

    # Store results
    models = {}
    results = {}

    # 2. Define functions to try
    def poly_func(x, *coeffs):
        return np.polyval(coeffs, x)

    def custom_func(x, L, k, x0):
        # Decreasing logistic-like function
        return L / (1 + np.exp(k * (x - x0)))
    
    def parametric_logistic(t, L, k, x0, theta):
        """
        Returns x(t), y(t) for a logistic curve rotated by theta.

        Parameters:
        - t: parameter (same as x in unrotated case)
        - L, k, x0: logistic parameters
        - theta: rotation angle in radians
        """
        y = L / (1 + np.exp(-k * (t - x0)))
        # Rotate around midpoint
        xc, yc = x0, L / 2
        x_shift = t - xc
        y_shift = y - yc
        x_rot = x_shift * np.cos(theta) - y_shift * np.sin(theta) + xc
        y_rot = x_shift * np.sin(theta) + y_shift * np.cos(theta) + yc
        return x_rot, y_rot
    
    def residuals(params, x_data, y_data):
        L, k, x0, theta = params
        x_model, y_model = parametric_logistic(x_data, L, k, x0, theta)
        return np.sqrt((x_model - x_data)**2 + (y_model - y_data)**2)

    def exp_func(x, a, b):
        return a * np.exp(b * x)

    def power_func(x, a, b):
        return a * np.power(x + 1e-6, b)  # avoid 0^b

    # --- Polynomial (try degrees 1 to 5)
    for deg in range(1, 3):
        coeffs = np.polyfit(x, y, deg)
        y_pred = np.polyval(coeffs, x)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        name = f"poly_deg_{deg}"
        results[name] = {'r2': r2,
                         'rmse': rmse,
                         'model': lambda x, c=coeffs: np.polyval(c, x),
                         'params': coeffs}

    # --- Exponential
    # try:
    #     popt, _ = curve_fit(exp_func, x, y, maxfev=10000)
    #     y_pred = exp_func(x, *popt)
    #     results['exponential'] = {'r2': r2_score(y, y_pred),
    #                               'rmse': np.sqrt(mean_squared_error(y, y_pred)),
    #                               'model': lambda x: exp_func(x, *popt),
    #                               'params': popt}
    # except Exception as e:
    #     print(f"Exponential fit failed: {e}")

    # --- Custom func (Decreasing Logistic-like function)
    # try:
    #     # Initial guess for parameters: L = max(y), k = 0.1, x0 = median(x)
    #     initial_guess = [max(y), 0.1, np.median(x)]
    #     popt, _ = curve_fit(custom_func, x, y, p0=initial_guess, maxfev=10000)
    #     L, k, x0 = popt
    #     y_pred = custom_func(x, *popt)
    #     results['custom'] = {'r2': r2_score(y, y_pred),
    #                          'rmse': np.sqrt(mean_squared_error(y, y_pred)),
    #                          'model': lambda x: custom_func(x, L, k, x0),
    #                          'params': popt}
    # except Exception as e:
    #     print(f"Custom function fit failed: {e}")

    # --- Rotated logistic function
    try:
        initial_guess = [max(y), 0.1, np.median(x), -0.2] # L, k, x0, theta
        res = least_squares(residuals, initial_guess, args=(x,y))
        L, k, x0, theta = res.x
        x_pred, y_pred = parametric_logistic(x, L, k, x0, theta)
        results["logistic"] = {'r2': r2_score(y, y_pred),
                             'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                             'model': lambda x: parametric_logistic(x, L, k, x0, theta),
                             'params': res.x}
    except Exception as e:
        print(f"Logistic function fit failed: {e}")

    # --- Power Law
    # try:
    #     popt, _ = curve_fit(power_func, x, y)
    #     y_pred = power_func(x, *popt)
    #     results['power'] = {'r2': r2_score(y, y_pred),
    #                         'rmse': np.sqrt(mean_squared_error(y, y_pred)),
    #                         'model': lambda x: power_func(x, *popt),
    #                         'params': popt}
    # except Exception as e:
    #     print(f"Power law fit failed: {e}")

    # --- Spline
    # spline = UnivariateSpline(x, y)
    # y_pred = spline(x)
    # results['spline'] = {'r2': r2_score(y, y_pred),
    #                      'rmse': np.sqrt(mean_squared_error(y, y_pred)),
    #                      'model': spline,
    #                      'params': popt}

    # 4. Find best model
    best_model = max(results.items(), key=lambda x: x[1]['r2'])
    print(f"Best model: {best_model[0]}, R² = {best_model[1]['r2']:.4f}, RMSE = {best_model[1]['rmse']:.4f}")

    # 5. Plot results
    plt.scatter(x, y, label='Original Data', color='black')
    x_dense = np.linspace(min(x), max(x), 500)

    for name, result in results.items():
        try:
            if name == "logistic":
                x_plot, y_plot = result['model'](x_dense)
                plt.plot(x_dense, y_plot, label=f"{name} (R²={result['r2']:.2f})")
            else:
                plt.plot(x_dense, result['model'](x_dense), label=f"{name} (R²={result['r2']:.2f})")
        except Exception as e:
            print(f"Error plotting {name}: {e}")
            continue

    plt.legend()
    plt.xlabel("x")
    plt.ylabel("zeta")
    plt.title("Model Comparison")
    plt.grid(True)
    plt.show()

    # Print the results
    print(f"Best model parameters: {best_model[1]['params']}")
    #print(list(val["zeta"].values()))
    #initial_points.append(list(val["zeta"].values())[0])