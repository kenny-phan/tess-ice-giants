import numpy as np
import matplotlib.pyplot as plt
import emcee
import warnings

from sklearn.mixture import GaussianMixture
from scipy import optimize, stats
from scipy.stats import norm, skew, gaussian_kde
from scipy.optimize import minimize_scalar, root_scalar
from scipy.signal import find_peaks

from frequency_analysis.wind_equations import frequency_wind_speed

# Log-likelihood function
def log_likelihood(phi, f_obs, f_err, model_eqn, sigma_eqn, freq_eqn):
    model = model_eqn(phi)
    expected = freq_eqn(phi, f_obs)
    sigma = sigma_eqn(phi, f_obs, f_err)
    return -0.5 * np.sum(((model - expected) / sigma)**2)

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
def run_mcmc(f_obs, f_err, model_eqn, sigma_eqn, freq_eqn, n_walkers=32, n_steps=5000):
    ndim = 1
    # Initialize walkers around 0 (equator)
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
# def run_mcmc(f_obs, model_eqn, freq_eqn, sigma=10.0, n_walkers=32, n_steps=5000):
#     ndim = 1
#     # Initialize walkers around 0 (equator)
#     initial_pos = np.random.uniform(0.1, np.pi/2 - 0.1, size=(n_walkers, ndim))

#     sampler = emcee.EnsembleSampler(n_walkers, ndim, log_probability, args=(f_obs, sigma, model_eqn, freq_eqn))
    
#     sampler.run_mcmc(initial_pos, n_steps, progress=True)

#     return sampler

# get latitude solutions for input peak frequency array, wind speed equation, and frequency eqn
def runanalysis(f, eqn, freq_eqn, n_components=2, plot=True):
    
    all_lats = []
    all_stdevs = []
            
    phi_deg_array = []
    
    for f_obs in f:
        sampler = run_mcmc(f_obs, eqn, freq_eqn)
        
        # Get the flattened samples
        samples = sampler.get_chain(discard=1000, flat=True)
        phi_samples = samples[:, 0]
        
        # Convert to degrees 
        phi_deg = np.degrees(phi_samples)
    
        phi_deg_array.append(phi_deg)
    
    latitudes, standard_devs = fit_gaussian(phi_deg_array, n_components=n_components, plot=plot)
    
    all_lats.append(latitudes)
    all_stdevs.append(standard_devs)

    return np.array(all_lats), np.array(all_stdevs)


# functions to determine the lower frequency bound of latitude solutions
# Residual
def residual(phi, f, wind_eqn, freq_eqn):
    return freq_eqn(phi, f) - wind_eqn(phi)

# Check if for a given f, residual = 0 has any solution in φ ∈ [a, b]
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
        minimum_frequency = find_minimum_frequency(wind_eqn, frequency_wind_speed(Req, Rp, P))
        minimum_frequencys.append(minimum_frequency)
    return np.array(minimum_frequencys)

def group_and_average(arr1, arr2, mean=True):
    """
    Groups elements of arr1 so that the number of groups matches the size of arr2.
    Returns an array of the averages of each group.
    """
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    
    len1 = len(arr1)
    len2 = len(arr2)

    if len2 == 0:
        raise ValueError("arr2 must have non-zero length.")
    if len1 < len2:
        raise ValueError("arr1 must be at least as long as arr2.")
    
    # Compute group size (may not be perfect division)
    group_size = len1 / len2

    result = []
    for i in range(len2):
        start = int(round(i * group_size))
        end = int(round((i + 1) * group_size))
        group = arr1[start:end]
        if mean:
            avg = np.mean(group) if len(group) > 0 else 0
        else: 
            avg = np.median(group) if len(group) > 0 else 0
        result.append(avg)

    print(f"{len(arr1) - len(result)*group_size} data points discarded")
    return np.array(result)

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


## POSTERIOR STATISTICAL CLASSIFICATION FUNCTIONS ##

# fit a gaussian to the mcmc posteriors
def fit_gaussian(phi_deg_array, n_components=None, plot=False):
    
    latitudes = []
    standard_devs = []

    for _, phi_deg in enumerate(phi_deg_array):
        bell = np.abs(phi_deg)
        data_reshaped = bell.reshape(-1, 1)

        if n_components is not None:
            gmm = GaussianMixture(n_components=n_components, random_state=42)
            gmm.fit(data_reshaped)
            means = gmm.means_.flatten()
            stds = np.sqrt(gmm.covariances_).flatten()
            
            if plot:
                # Plotting
                x = np.linspace(bell.min(), bell.max(), 1000).reshape(-1, 1)
                logprob = gmm.score_samples(x)
                pdf = np.exp(logprob)
                plt.plot(x, pdf, label=f"{n_components}-Gaussian GMM", color="red")
        
        else:
            mean = np.mean(bell)
            std = np.std(bell)
            means = np.array([mean])
            stds = np.array([std])
            
            if plot: 
                # Plotting
                x = np.linspace(bell.min(), bell.max(), 1000)
                pdf = norm.pdf(x, loc=mean, scale=std)
                plt.plot(x, pdf, label="Single Gaussian", color="blue")

        latitudes.append(means)
        standard_devs.append(stds)

        if plot:
            # Histogram
            plt.hist(bell, bins=50, density=True, alpha=0.5, label="Data")
            plt.legend()
            distribution_type = ["Unimodal", "Bimodal", "Trimodal"]
            plt.title(f"{distribution_type[n_components - 1]} Gaussian Fit")
            plt.xlabel("Value")
            plt.ylabel("Density")
            plt.ticklabel_format(style='plain', axis='x')

            plt.show()
    
            print("Means:", means)
            print("Standard Deviations:", stds)

    return latitudes, standard_devs


def fit_truncated_normal(x, b, mu0=None, sigma0=None):
    x = np.asarray(x)
    if mu0 is None: mu0 = np.mean(x)
    if sigma0 is None: sigma0 = np.std(x, ddof=1)

    def neg_loglike(params):
        mu, log_sigma = params
        sigma = np.exp(log_sigma)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            z = (x - mu) / sigma
            log_pdf = stats.norm.logpdf(z) - np.log(sigma)
            log_norm_const = stats.norm.logcdf((b - mu) / sigma)

        return -np.sum(log_pdf - log_norm_const)

    res = optimize.minimize(
        neg_loglike,
        x0=[mu0, np.log(sigma0)],
        method="L-BFGS-B"
    )
    mu_hat, sigma_hat = res.x[0], np.exp(res.x[1])
    return mu_hat, sigma_hat, res



def credible_interval(samples, ci=0.68):
    """
    Compute a credible interval for a (possibly skewed) distribution
    using quantiles. Works for any posterior shape.

    Parameters
    ----------
    samples : array-like
        Posterior samples.
    ci : float
        Credible interval (e.g., 0.68, 0.90, 0.95).

    Returns
    -------
    lower : float
        Lower credible bound.
    median : float
        Median of the distribution.
    upper : float
        Upper credible bound.
    """
    samples = np.asarray(samples)

    alpha = (1 - ci) / 2
    lower = np.quantile(samples, alpha)
    median = np.quantile(samples, 0.5)
    upper = np.quantile(samples, 1 - alpha)

    return lower, median, upper

def detect_truncation(samples, boundaries=[0, 90], delta_loglike=5):
    """
    Return True if posterior is significantly better fit by a truncated-normal 
    than by a standard normal.
    delta_loglike : threshold difference for significance (in log-evidence units)
    """
    samples = np.asarray(samples)
    low, high = boundaries

    # Fit full (untruncated) Gaussian
    mu_full = np.mean(samples)
    sigma_full = np.std(samples)

    ll_full = np.sum(stats.norm.logpdf(samples, mu_full, sigma_full))

    # Fit truncated normal to BOTH boundaries
    mu_t, sigma_t, _ = fit_truncated_normal(samples, b=low,  mu0=mu_full, sigma0=sigma_full)
    ll_low = np.sum(stats.truncnorm.logpdf(
        samples, (low - mu_t)/sigma_t, (high - mu_t)/sigma_t, loc=mu_t, scale=sigma_t
    ))

    mu_t2, sigma_t2, _ = fit_truncated_normal(samples, b=high, mu0=mu_full, sigma0=sigma_full)
    ll_high = np.sum(stats.truncnorm.logpdf(
        samples, (low - mu_t2)/sigma_t2, (high - mu_t2)/sigma_t2, loc=mu_t2, scale=sigma_t2
    ))

    ll_trunc = max(ll_low, ll_high)

    # If truncated log-likelihood is much higher → it's truncated
    return (ll_trunc - ll_full) > delta_loglike, ll_low, ll_high


def detect_bimodality(samples, min_prominence=0.005):
    samples = np.asarray(samples, dtype=float).ravel()

    kde = gaussian_kde(samples)
    xs = np.linspace(samples.min(), samples.max(), 2000)
    ys = kde(xs)

    # find all peaks
    peaks, _ = find_peaks(ys, prominence=min_prominence * np.max(ys))

    if len(peaks) < 2:
        return "unimodal"
    
    return "bimodal"


def classify_posterior(samples, boundaries=[0, 90], verbose=True):  
    samples = np.asarray(samples).ravel()

    classification = detect_bimodality(samples)

    if classification == "unimodal":
        s = skew(samples)
        mean, std = np.mean(samples), np.std(samples)
        mu_hat_0, sigma_hat_0, _ = fit_truncated_normal(samples, mu0=mean, sigma0=std, b=boundaries[0])
        mu_hat_1, sigma_hat_1, _ = fit_truncated_normal(samples, mu0=mean, sigma0=std, b=boundaries[1])

        truc_bool, ll_low, ll_high = detect_truncation(samples, boundaries)
        debug_print(verbose, msg=f"Skewness: {s:.3f}, Mean: {mean:.3f}, Std: {std:.3f}, Mu0: {mu_hat_0:.3f}, Sigma0: {sigma_hat_0:.3f}, Mu1: {mu_hat_1:.3f}, Sigma1: {sigma_hat_1:.3f}")
        
        # Skewness dominates
        if abs(s) > 1: #or abs(quant_asym) > 0.5
            lower, median, upper = credible_interval(samples, ci=0.68)
            lower_bound, upper_bound = np.abs(median - lower), np.abs(upper - median)
            classification = "Skewed"
            return classification, median, [lower_bound, upper_bound]
        
        if truc_bool:
            classification = "Truncated Gaussian"
            # choose which boundary is better fit
            if ll_low > ll_high:
                return classification, mu_hat_0, sigma_hat_0
            else:
                return classification, mu_hat_1, sigma_hat_1

        else: 
            classification = "Gaussian"
            lat, std = fit_gaussian([samples], n_components=1, plot=False)
            return classification, lat[0][0], std[0][0]

    else:
        classification = "Bimodal"
        lat, std = fit_gaussian([samples], n_components=2, plot=False)
        return classification, lat[0], std[0]
    
    
#distribution is one wind equation's latitude posterior samples (e.g. uranus s44 sromovsky2012N)
def parse_classifications(distribution, plot=True, verbose=False):
    all_means = []
    all_stds = []

    for dist in distribution:
        dist = np.asarray(dist, dtype=float).ravel()   # ensure numeric
        
        classification_result = classify_posterior(dist)
        debug_print(verbose, msg=f"Classification Result: {classification_result[0]}")

        all_means.append(classification_result[1])
        all_stds.append(classification_result[2])

        if plot:
            plt.hist(dist, bins=30, density=True)
            plt.show()

    return all_means, all_stds

    
# phi_distributions_list: one sector's data
def fit_all_distributions(phi_distributions_list, wind_eqn_strings, plot=False, print_table=True, verbose=False):
    all_latitudes = []
    all_standard_devs = []

    for i, phi_distributions in enumerate(phi_distributions_list):
        print(f"Processing Wind Equation: {wind_eqn_strings[i]}")
        latitudes, standard_devs = parse_classifications(phi_distributions, plot=plot, verbose=verbose)
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

def debug_print(verbose, msg=""):
    if verbose == True:
        print(msg)