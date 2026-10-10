import emcee
import os

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm, truncnorm
from scipy.optimize import minimize_scalar, root_scalar

from bootstrap import classify_posterior
from fullsector import debug_print
from wind_equations import RHS, U_PHI, sigma
from figures import dpi

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
def log_probability(phi, f_obs, sigma_f, model_eqn, sigma_eqn, freq_eqn):
    lp = log_prior(phi)
    if not np.isfinite(lp):
        return -np.inf  
    
    return lp + log_likelihood(phi, f_obs, sigma_f, model_eqn, sigma_eqn, freq_eqn)

# Run the sampler
def mcmc(f_obs, f_err, model_eqn, sigma_eqn, freq_eqn, n_walkers=32, n_steps=5000):

    ndim = 1
    # Initialize walkers anywhere from 0 to 90 degrees
    initial_pos = np.random.uniform(0.1, np.pi/2 - 0.1, size=(n_walkers, ndim))

    sampler = emcee.EnsembleSampler(n_walkers, 
                                    ndim, 
                                    log_probability, 
                                    args=(f_obs, f_err, 
                                        model_eqn, sigma_eqn, 
                                        freq_eqn))
    
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
                          truncbound=[0, 90], 
                          floor_frac=0.01, 
                          promfrac=0.01,
                          mask=False,
                          n_bins=None, 
                          sds="sector",
                          figdir=None, 
                          verbose=False):
    all_means = []
    all_stds = []

    for i, dist in enumerate(distribution):
        dist = np.asarray(dist, dtype=float).ravel()   # ensure numeric
        
        classification_result = classify_posterior(dist, 
                                                   truncbound=truncbound, 
                                                   n_bins=n_bins, 
                                                   floor_frac=floor_frac, 
                                                   promfrac=promfrac,
                                                   mask=mask,
                                                   verbose=verbose
                                                #    boundaries=boundaries, 
                                                #    min_prominence=min_prominence,
                                                #    allow_skew_truc=allow_skew_truc, 
                                                #    skew_threshold=skew_threshold
                                                   )
        debug_print(verbose, f"Classification Result: {classification_result[0]}")

        all_means.append(classification_result[1])
        all_stds.append(classification_result[2])
        
        if figdir is not None:
            classification_type = classification_result[0]
            pdfx, pdfy = classification_result[4], classification_result[5]
            save_dir = os.path.join(figdir, "lat_posteriors")
            os.makedirs(save_dir, exist_ok=True)

            # figsize + DPI controls the output size; without this, every
            # iteration reuses whatever figure pyplot has open.
            fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)

            ax.hist(dist, bins=30, alpha=0.7, label='Data', density=False)
            ax.plot(pdfx, pdfy, color='red', label='Fit')

            ax.set_xlabel('Value')
            ax.set_ylabel('Count')
            ax.set_title(f'Classification: {classification_type}')
            ax.legend()

            # Include a stable per-figure identifier. `sds` alone is not unique
            # across the loop, so add the index and a distinguishing tag.
            # If you have per-distribution labels, use those instead.
            fname = f"{sds}_{i}_{classification_type}.png"
            fpath = os.path.join(save_dir, fname)
            fig.savefig(fpath, bbox_inches='tight')

            # Critical: free the figure so the next iteration starts clean.
            plt.close(fig)
                    
            # Plot PDF based on classification type
            # if classification_type == "Gaussian":
            #     pdf = norm.pdf(x, mean_val, param_val)
            #     plt.plot(x, pdf, 'r-', linewidth=2, label=f'Gaussian: μ={mean_val:.3f}, σ={param_val:.3f}')
            
            # elif classification_type == "Truncated Gaussian":
            #     a, b = (truncbound[0] - mean_val) / param_val, (truncbound[1] - mean_val) / param_val
            #     pdf = truncnorm.pdf(x, a, b, loc=mean_val, scale=param_val)
            #     plt.plot(x, pdf, 'r-', linewidth=2, label=f'Truncated Gaussian: μ={mean_val:.3f}, σ={param_val:.3f}')
            
            # elif classification_type == "Skewed":
            #     plt.axvline(mean_val, color='r', linestyle='--', linewidth=2, 
            #             label=f'Median={mean_val:.3f}, 68% CI=[{mean_val-param_val[0]:.3f}, {mean_val+param_val[1]:.3f}]')
            
            # elif classification_type == "Bimodal":
            #     pdf = 0.5 * norm.pdf(x, mean_val[0], param_val[0]) + 0.5 * norm.pdf(x, mean_val[1], param_val[1])
            #     plt.plot(x, pdf, 'r-', linewidth=2, 
            #             label=f'Bimodal: μ1={mean_val[0]:.3f}, σ1={param_val[0]:.3f} | μ2={mean_val[1]:.3f}, σ2={param_val[1]:.3f}')
        

    return all_means, all_stds

    
# phi_distributions_list: one sector's data
def fit_all_distributions(phi_distributions_list, 
                          wind_eqn_strings, 
                          print_table=True,
                          truncbound=[0,90], 
                          n_bins=None,
                          floor_frac=0.01, 
                          promfrac=0.01,
                          mask=False,
                          sds="sector",
                          figdir=None, 
                          verbose=False):
    all_latitudes = []
    all_standard_devs = []

    for i, phi_distributions in enumerate(phi_distributions_list):
        print(f"Processing Wind Equation: {wind_eqn_strings[i]}")
        latitudes, standard_devs = parse_classifications(phi_distributions, 
                                                         truncbound=truncbound,
                                                         n_bins=n_bins,
                                                         floor_frac=floor_frac,
                                                         promfrac=promfrac,
                                                         sds=sds + f"_dist{i}",
                                                         mask=mask, 
                                                         figdir=figdir,  
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

def multiply_nested(std, mean_squared):
    """Recursively multiply nested structures."""
    if isinstance(std, (int, float)):
        return std * mean_squared
    elif isinstance(std, list):
        return [multiply_nested(s, mean_squared) for s in std]
    else:
        # Handle NumPy arrays
        return std * mean_squared

def is_homogeneous(arr):
    """Check if all elements are the same type."""
    # if not arr:
    #     return True
    if len(arr) == 0:
        return True
    return all(type(elem) == type(arr[0]) for elem in arr)

def save_mcmc(wind_eqns, wind_eqn_errs, cluster_arr, 
              Re, Rp, P, Re_err, Rp_err, P_err, 
              wind_eqn_strings, sector_data_string, save_dir, reperrs=None,
              min_freq_threshold=0.5, n_steps=5000):

    os.makedirs(save_dir, exist_ok=True)
    
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

        print('matched means', cluster_arr['matched_means'])
        print('matched stds', cluster_arr['matched_stds'])

        means = np.array(cluster_arr['matched_means'])
        means_filtered = 1 / means[means < period_limit]

        clust_stds = cluster_arr['matched_stds']
        if is_homogeneous(clust_stds):
            stds = np.array(cluster_arr['matched_stds'])
            stds_filtered_periods = stds[means < period_limit]
            stds_filtered = stds_filtered_periods * (means_filtered[:, None] ** 2)

            # stds_filtered = stds_filtered_periods * (means_filtered**2)

        else:
            print('not homogeneous')
            stds = clust_stds
            stds_filtered_periods = [std for std, mean in zip(stds, means) if mean < period_limit]
            stds_filtered = [multiply_nested(std, mean**2) 
                             for std, mean in zip(stds_filtered_periods, means_filtered)]
        # default units of 'matched means' is days
        # take only the peaks that are below the period limit, convert to 1/days

        # # standard deviations, default units of days

        # # uncertainty in frequency is related to uncertainty in period by sigma_f = (1/P^2) * sigma_P, where P is the period
        # means_filtered = frequencies[frequencies > min_freq]
        # stds_filtered = frequency_errs[frequencies > min_freq]

        print("Processing wind equation:", wind_eqn_strings[i])
        for f_obs, f_err in zip(means_filtered, stds_filtered):
            # f_err = np.array(f_err, dtype=float)  # ensure numeric
            print("Frequency, error:", f_obs, f_err)

            if isinstance(f_err, (list, tuple, np.ndarray)) and len(f_err) == 2:
                print("Skewed distribution. Taking lower chain")
                sampler1 = mcmc(f_obs, f_err[0], model_eqn, sigma_eqn, freq_eqn, n_steps=n_steps)
                samples1 = sampler1.get_chain(discard=1000, flat=True)
                phi_samples1 = samples1[:, 0]
                phi_deg1 = np.array(np.degrees(phi_samples1))

                print("Median latitude (deg):", np.median(phi_deg1))
                print("Std", np.std(phi_deg1))

                print("Taking upper chain")

                sampler2 = mcmc(f_obs, f_err[1], model_eqn, sigma_eqn, freq_eqn, n_steps=n_steps)
                samples2 = sampler2.get_chain(discard=1000, flat=True)
                phi_samples2 = samples2[:, 0]
                phi_deg2 = np.array(np.degrees(phi_samples2))
                print("Median latitude (deg):", np.median(phi_deg2))
                print("Std", np.std(phi_deg2))


                threshold = (np.mean(phi_deg1) + np.mean(phi_deg2)) / 2
                phi_deg = np.concatenate([
                    phi_deg1[phi_deg1 < threshold],
                    phi_deg2[phi_deg2 > threshold]
                ])

                print("Total Median latitude (deg):", np.median(phi_deg))
                print("Total Std", np.std(phi_deg))

            else:
                sampler = mcmc(f_obs, f_err, model_eqn, sigma_eqn, freq_eqn, n_steps=n_steps)

                samples = sampler.get_chain(discard=1000, flat=True)
                phi_samples = samples[:, 0]

                # Convert to degrees 
                phi_deg = np.array(np.degrees(phi_samples))

                print("Median latitude (deg):", np.median(phi_deg))
                print("Std", np.std(phi_deg))
            phi_arr.append(phi_deg)
            print(f"appended phi_deg of median {np.median(phi_deg)}")
        phi_super_arr.append(phi_arr)
        i += 1
    print("saving to: ", save_dir + f'{sector_data_string}_phi_distributions.npz')
    np.savez(save_dir + f'{sector_data_string}_phi_distributions.npz', 
             wind_eqn_strings=wind_eqn_strings, phi_distributions=np.array(phi_super_arr, dtype=object))