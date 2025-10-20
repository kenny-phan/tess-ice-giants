import numpy as np
import matplotlib.pyplot as plt

import emcee

from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from scipy.optimize import minimize_scalar, root_scalar

# Log-likelihood function
def log_likelihood(phi, f_obs, sigma, model_eqn, freq_eqn):
    model = model_eqn(phi)
    expected = freq_eqn(phi, f_obs)
    return -0.5 * np.sum(((model - expected) / sigma)**2)

# Log-prior (uniform in latitude range)
def log_prior(phi):
    if 0 < phi < np.pi/2:  # latitude in radians
        return 0.0
    return -np.inf

# Full log-probability
def log_probability(phi, f_obs, sigma, model_eqn, freq_eqn):
    lp = log_prior(phi)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(phi, f_obs, sigma, model_eqn, freq_eqn)

# Run the sampler
def run_mcmc(f_obs, model_eqn, freq_eqn, sigma=10.0, n_walkers=32, n_steps=5000):
    ndim = 1
    # Initialize walkers around 0 (equator)
    initial_pos = np.random.uniform(0.1, np.pi/2 - 0.1, size=(n_walkers, ndim))

    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_probability, args=(f_obs, sigma, model_eqn, freq_eqn))
    
    sampler.run_mcmc(initial_pos, n_steps, progress=True)

    return sampler

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
                plt.plot(x, pdf, label="2-Gaussian GMM", color="red")
        
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
            plt.show()
    
            print("Means:", means)
            print("Standard Deviations:", stds)

    return latitudes, standard_devs

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
