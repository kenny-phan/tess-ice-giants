import emcee

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm, truncnorm
from scipy.optimize import minimize_scalar, root_scalar

from bootstrap import classify_posterior
from fullsector import debug_print
from wind_equations import RHS, U_PHI, sigma

# Log-likelihood function
# phi radians, f_obs 1/days, f_err 1/days, model_eqn m/s
def log_likelihood(phi, f_obs, f_err, model_eqn, sigma_eqn, freq_eqn):
    model = model_eqn(phi)
    data = freq_eqn(phi, f_obs)
    sigma_func = sigma_eqn
    return -0.5 * np.sum(((data - model) / sigma_func(phi, f_obs, f_err))**2)

# Log-prior (uniform in latitude range)
def log_prior(phi):
    if 0 < phi < np.pi/2:  # latitude in radians
        return 0.0
    return -np.inf

# Full log-probability
def log_probability(phi, f_obs, f_err, model_eqn, sigma_eqn, freq_eqn):
    lp = log_prior(phi)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(phi, f_obs, f_err, model_eqn, sigma_eqn, freq_eqn)

# Run the sampler
def mcmc(f_obs, f_err, model_eqn, sigma_eqn, freq_eqn, n_walkers=32, n_steps=5000):
    ndim = 1
    # Initialize walkers anywhere from 0 to 90 degrees
    initial_pos = np.random.uniform(0.1, np.pi/2 - 0.1, size=(n_walkers, ndim))

    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_probability, args=(f_obs, f_err, model_eqn, sigma_eqn, freq_eqn))
    
    sampler.run_mcmc(initial_pos, n_steps, progress=True)

    return sampler

# OLD PROGRAMS PRIOR TO ERROR PROPAGATION
# # Log-likelihood function
# def log_likelihood(phi, f_obs, sigma, model_eqn, freq_eqn):
#     model = model_eqn(phi)
#     expected = freq_eqn(phi, f_obs)
#     return -0.5 * np.sum(((model - expected) / sigma)**2)

# # Log-prior (uniform in latitude range)
# def log_prior(phi):
#     if 0 < phi < np.pi/2:  # latitude in radians
#         return 0.0
#     return -np.inf

# # Full log-probability
# def log_probability(phi, f_obs, sigma, model_eqn, freq_eqn):
#     lp = log_prior(phi)
#     if not np.isfinite(lp):
#         return -np.inf
#     return lp + log_likelihood(phi, f_obs, sigma, model_eqn, freq_eqn)

# # Run the sampler
# def mcmc(f_obs, model_eqn, freq_eqn, sigma=10.0, n_walkers=32, n_steps=5000):
#     ndim = 1
#     # Initialize walkers around 0 (equator)
#     initial_pos = np.random.uniform(0.1, np.pi/2 - 0.1, size=(n_walkers, ndim))

#     sampler = emcee.EnsembleSampler(n_walkers, ndim, log_probability, args=(f_obs, sigma, model_eqn, freq_eqn))
    
#     sampler.run_mcmc(initial_pos, n_steps, progress=True)

#     return sampler

# get latitude solutions for input peak frequency array, wind speed equation, and frequency eqn
# def runanalysis(f, eqn, freq_eqn, Re, Rp, P, sigma_Re, sigma_Rp, sigma_P, n_components=2, n_steps=5000, plot=True):
    
#     all_lats = []
#     all_stdevs = []
            
#     phi_deg_array = []
    
#     for f_obs in f:
#         sampler = mcmc(f_obs, eqn, freq_eqn, n_steps=n_steps)
        
#         # Get the flattened samples
#         samples = sampler.get_chain(discard=1000, flat=True)
#         phi_samples = samples[:, 0]
        
#         # Convert to degrees 
#         phi_deg = np.degrees(phi_samples)
    
#         phi_deg_array.append(phi_deg)
    
#     latitudes, standard_devs = fit_gaussian(phi_deg_array, n_components=n_components, plot=plot)
    
#     all_lats.append(latitudes)
#     all_stdevs.append(standard_devs)

#     return np.array(all_lats), np.array(all_stdevs)

def solve_intersection_at_phi(wind_eqn, freq_eqn, bounds=(0.01, 2), phi=0.0):
    """
    Solve for f such that wind_eqn(0) = freq_eqn(f, 0).
    """
    target = wind_eqn(phi)  # fixed value at phi=0

    def func(f):
        return freq_eqn(f, phi) - target

    result = root_scalar(func, bracket=bounds, method='brentq')
    if result.converged:
        print(f"Intersection at phi=0: f = {result.root:.6f}")
        return result.root
    else:
        raise RuntimeError("No intersection found in the given bounds")
    
    
#distribution is one wind equation's latitude posterior samples (e.g. uranus s44 sromovsky2012N)
def parse_classifications(distribution, 
                          boundaries=[0, 90], 
                          min_prominence=0.01,
                          plot=True, 
                          verbose=False, 
                          allow_skew_truc=True):
    all_means = []
    all_stds = []

    for dist in distribution:
        dist = np.asarray(dist, dtype=float).ravel()   # ensure numeric
        
        classification_result = classify_posterior(dist, 
                                                   boundaries=boundaries, 
                                                   min_prominence=min_prominence,
                                                   allow_skew_truc=allow_skew_truc)
        debug_print(verbose, f"Classification Result: {classification_result[0]}")

        all_means.append(classification_result[1])
        all_stds.append(classification_result[2])

        if plot:
            classification_type = classification_result[0]
            mean_val = classification_result[1]
            param_val = classification_result[2]
            
            # Plot histogram
            x = np.linspace(dist.min(), dist.max(), 200)
            plt.hist(dist, bins=30, density=True, alpha=0.7, label='Data')
            
            # Plot PDF based on classification type
            if classification_type == "Gaussian":
                pdf = norm.pdf(x, mean_val, param_val)
                plt.plot(x, pdf, 'r-', linewidth=2, label=f'Gaussian: μ={mean_val:.3f}, σ={param_val:.3f}')
            
            elif classification_type == "Truncated Gaussian":
                a, b = (boundaries[0] - mean_val) / param_val, (boundaries[1] - mean_val) / param_val
                pdf = truncnorm.pdf(x, a, b, loc=mean_val, scale=param_val)
                plt.plot(x, pdf, 'r-', linewidth=2, label=f'Truncated Gaussian: μ={mean_val:.3f}, σ={param_val:.3f}')
            
            elif classification_type == "Skewed":
                plt.axvline(mean_val, color='r', linestyle='--', linewidth=2, 
                        label=f'Median={mean_val:.3f}, 68% CI=[{mean_val-param_val[0]:.3f}, {mean_val+param_val[1]:.3f}]')
            
            elif classification_type == "Bimodal":
                pdf = 0.5 * norm.pdf(x, mean_val[0], param_val[0]) + 0.5 * norm.pdf(x, mean_val[1], param_val[1])
                plt.plot(x, pdf, 'r-', linewidth=2, 
                        label=f'Bimodal: μ1={mean_val[0]:.3f}, σ1={param_val[0]:.3f} | μ2={mean_val[1]:.3f}, σ2={param_val[1]:.3f}')
            
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.title(f'Classification: {classification_type}')
            plt.legend()
            plt.show()


    return all_means, all_stds

    
# phi_distributions_list: one sector's data
def fit_all_distributions(phi_distributions_list, 
                          wind_eqn_strings, 
                          min_prominence=0.01,
                          plot=False, 
                          print_table=True, 
                          verbose=False):
    all_latitudes = []
    all_standard_devs = []

    for i, phi_distributions in enumerate(phi_distributions_list):
        print(f"Processing Wind Equation: {wind_eqn_strings[i]}")
        latitudes, standard_devs = parse_classifications(phi_distributions, 
                                                         min_prominence=min_prominence,
                                                         plot=plot, 
                                                         verbose=verbose)
        all_latitudes.append(latitudes)
        all_standard_devs.append(standard_devs)
        
    if print_table:

        n_eqns = len(wind_eqn_strings)
        n_rows = len(all_latitudes[0])   # solutions per eqn

        # --- Header ---
        header = " ".join([f"& Eqn {i+1}" for i in range(n_eqns)])
        print(header)
        
        # --- Each row ---
        for k in range(n_rows):
            row_entries = []
            for i in range(n_eqns):
                lat  = all_latitudes[i][k]
                std  = all_standard_devs[i][k]
                lat = np.atleast_1d(lat)
                std = np.atleast_1d(std)
                if len(lat) == 1 and len(std) == 1:
                    row_entries.append(f"{lat[0]:.2f} ± {std[0]:.2f}")
                elif len(lat) == 2 and len(std) == 2:
                    row_entries.append(f"{lat[0]:.2f} ± {std[0]:.2f}, {lat[1]:.2f} ± {std[1]:.2f}")
                elif len(lat) == 1 and len(std) == 2:
                    row_entries.append(f"{lat[0]:.2f}_{{-{std[0]:.2f}}}^{{+{std[1]:.2f}}}")
            
            print("& " + " & ".join(row_entries) + " \\\\")

    return all_latitudes, all_standard_devs

def sigma_f_to_period(sigma_f, frequency):
    return sigma_f / frequency**2

# functions to determine the lower frequency bound of latitude solutions
# Residual
def residual(phi, f, wind_eqn, freq_eqn):
    return freq_eqn(phi, f) - wind_eqn(phi)

# Check if for a given f, residual = 0 has any solution in in phi between range
def has_root(f, wind_eqn, freq_eqn, phi_range=(-np.pi/2, np.pi/2), num_points=1000):
    phi_vals = np.linspace(phi_range[0], phi_range[1], num_points)
    res_vals = residual(phi_vals, f, wind_eqn, freq_eqn)
    
    # Check for a sign change (indicates root crossing)
    return np.any(np.diff(np.sign(res_vals)))

# Objective function: return 0 if root exists, else large penalty
def objective(f, wind_eqn, freq_eqn):
    return f if has_root(f, wind_eqn, freq_eqn) else np.inf

    # Use scalar minimization (bounded search)
def find_minimum_frequency(wind_eqn, freq_eqn, bounds=(0.01, 2)):
    result = minimize_scalar(
        lambda f: objective(f, wind_eqn, freq_eqn),
        bounds=bounds,
        method='bounded'
    )
    if result.success and np.isfinite(result.fun):
        print(f"Minimum f with at least one intersection: {result.x:.6f}")
    else:
        print("No intersection found in the given f range.")

    return result.x


def get_minimum_frequency_arr(wind_eqns, Req, Rp, P):
    minimum_frequencys = []
    for wind_eqn in wind_eqns:
        minimum_frequency = find_minimum_frequency(wind_eqn, RHS(Req, Rp, P))
        minimum_frequencys.append(minimum_frequency)
    return np.array(minimum_frequencys)

def save_mcmc(wind_eqns, wind_eqn_errs, cluster_arr, 
              Re, Rp, P, Re_err, Rp_err, P_err, 
              wind_eqn_strings, sector_data_string, root, reperrs=None,
              min_freq_threshold=0.5, n_steps=5000):
    
    min_freq_arr = get_minimum_frequency_arr(wind_eqns, Re, Rp, P)
    freq_eqn = RHS(Re, Rp, P) 

    phi_super_arr = []
    i = 0
    for wind_eqn, wind_eqn_err in zip(wind_eqns, wind_eqn_errs):
        model_eqn = U_PHI(wind_eqn)

        if reperrs is not None:
            reperr = reperrs[i]
            sigma_eqn = sigma(Re, Rp, P, Re_err, Rp_err, P_err, wind_eqn_err, reperr=reperr)
        else:   
            sigma_eqn = sigma(Re, Rp, P, Re_err, Rp_err, P_err, wind_eqn_err)

        # if the minumum frequency is extremely low, use the next lowest frequency that is above the threshold
        if (min_freq_arr[i] > min_freq_threshold):
            min_freq = min_freq_arr[i] 
        else: 
            min_freq = min_freq_arr[np.argmin(min_freq_arr[min_freq_arr < min_freq_threshold])]

        print(f"Using minimum frequency of {min_freq} for wind equation {wind_eqn_strings[i]}")
        period_limit = 1 / min_freq # maximum period in days

        phi_arr = []

        # default units of 'matched means' is days
        # take only the peaks that are below the period limit, convert to 1/days
        means_filtered = 1 / cluster_arr['matched_means'][cluster_arr['matched_means'] < period_limit]

        # # standard deviations, default units of days
        stds_filtered_periods = cluster_arr['matched_stds'][cluster_arr['matched_means'] < period_limit]

        # # uncertainty in frequency is related to uncertainty in period by sigma_f = (1/P^2) * sigma_P, where P is the period
        stds_filtered = stds_filtered_periods * (means_filtered**2)
        # means_filtered = frequencies[frequencies > min_freq]
        # stds_filtered = frequency_errs[frequencies > min_freq]

        print("Processing wind equation:", wind_eqn_strings[i])
        for f_obs, f_err in zip(means_filtered, stds_filtered):
            print("Frequency, error:", f_obs, f_err)
            sampler = mcmc(f_obs, f_err, model_eqn, sigma_eqn, freq_eqn, n_steps=n_steps)

            samples = sampler.get_chain(discard=1000, flat=True)
            phi_samples = samples[:, 0]

            # Convert to degrees 
            phi_deg = np.array(np.degrees(phi_samples))

            print("Median latitude (deg):", np.median(phi_deg))
            phi_arr.append(phi_deg)
        phi_super_arr.append(phi_arr)
        i += 1

    np.savez(root + f'{sector_data_string}_phi_distributions.npz', 
             wind_eqn_strings=wind_eqn_strings, phi_distributions=np.array(phi_super_arr, dtype=object))