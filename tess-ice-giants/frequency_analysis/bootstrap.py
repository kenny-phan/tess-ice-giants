import os
import warnings

import numpy as np
import matplotlib.pyplot as plt

from astropy.timeseries import LombScargle
from scipy import optimize, stats
from scipy.signal import find_peaks
from scipy.stats import norm, skew, gaussian_kde, truncnorm, skewnorm
from scipy.optimize import curve_fit
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

from fullsector import debug_print, get_peak_frequencies
from figures import plot_colors_rgb, dpi

warnings.filterwarnings('ignore') 


def fit_gaussian(x, y):
    
    # Convert to numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Define Gaussian function for fitting
    def gaussian(x, amp, mean, std):
        return amp * np.exp(-0.5 * ((x - mean) / std) ** 2)
    
    # Initial guesses
    # Mean: weighted average
    mean_guess = np.sum(x * y) / np.sum(y)
    # Std: weighted standard deviation
    std_guess = np.sqrt(np.sum(y * (x - mean_guess) ** 2) / np.sum(y))
    # Amplitude: max of y
    amp_guess = np.max(y)
    
    # Fit the Gaussian
    try:
        popt, pcov = curve_fit(
            gaussian, x, y,
            p0=[amp_guess, mean_guess, std_guess],
            maxfev=10000
        )
        amp_fit, mean_fit, std_fit = popt
    except RuntimeError:
        # Fall back to weighted estimates if fitting fails
        mean_fit = mean_guess
        std_fit = std_guess
        amp_fit = amp_guess
    
    # Generate fitted PDF with more points for smoothness
    x_fit = np.linspace(x.min(), x.max(), len(x))
    y_fit = gaussian(x_fit, amp_fit, mean_fit, std_fit)
    
    return mean_fit, std_fit, x_fit, y_fit    

def fit_bimodal_gaussian(x, y, n_restarts=10):
    
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Estimate grid spacing for proper normalization
    dx = np.mean(np.diff(x)) if len(x) > 1 else 1.0
    
    # Normalize y into a proper PDF so weights have physical meaning
    area = np.sum(y) * dx
    if area > 0:
        y_pdf = y / area
    else:
        y_pdf = y.copy()
    
    # Define bimodal Gaussian (unnormalized form for fitting)
    def bimodal(x, amp1, mu1, sig1, amp2, mu2, sig2):
        g1 = amp1 * np.exp(-0.5 * ((x - mu1) / sig1) ** 2)
        g2 = amp2 * np.exp(-0.5 * ((x - mu2) / sig2) ** 2)
        return g1 + g2
    
    # Helper: weighted mean and std of the whole distribution
    w_mean = np.sum(x * y_pdf) * dx
    w_std = np.sqrt(np.sum(y_pdf * (x - w_mean) ** 2) * dx)
    amp_guess = np.max(y_pdf)
    
    # Initial guess: split the distribution at the weighted mean
    # and estimate stats of each half
    left_mask = x < w_mean
    right_mask = x >= w_mean
    
    def half_stats(mask):
        if not np.any(mask):
            return w_mean, w_std
        xs, ys = x[mask], y_pdf[mask]
        m = np.sum(xs * ys) / np.sum(ys) if np.sum(ys) > 0 else w_mean
        s = np.sqrt(np.sum(ys * (xs - m) ** 2) / np.sum(ys)) if np.sum(ys) > 0 else w_std
        return m, max(s, dx)  # avoid zero std
    
    mu1_g, sig1_g = half_stats(left_mask)
    mu2_g, sig2_g = half_stats(right_mask)
    
    # Ensure mu1 < mu2 for consistent sorting
    if mu1_g > mu2_g:
        mu1_g, mu2_g = mu2_g, mu1_g
        sig1_g, sig2_g = sig2_g, sig1_g
    
    # Amplitude guesses: fraction of total mass on each side
    frac_left = np.sum(y_pdf[left_mask]) * dx if np.any(left_mask) else 0.5
    amp1_g = max(frac_left, 0.1) * amp_guess / (sig1_g * np.sqrt(2 * np.pi))
    amp2_g = max(1 - frac_left, 0.1) * amp_guess / (sig2_g * np.sqrt(2 * np.pi))
    
    p0_base = [amp1_g, mu1_g, sig1_g, amp2_g, mu2_g, sig2_g]
    
    # Bounds: amplitudes >= 0, stds > 0
    x_min, x_max = x.min(), x.max()
    x_range = x_max - x_min
    lower = [0,          x_min, 1e-6, 0,          x_min, 1e-6]
    upper = [np.inf,     x_max, x_range, np.inf, x_max, x_range]
    
    best_popt = None
    best_resid = np.inf
    
    rng = np.random.default_rng(42)
    for i in range(n_restarts):
        if i == 0:
            p0 = p0_base
        else:
            # Random perturbations around the base guess
            p0 = np.array(p0_base, dtype=float)
            p0[1] += rng.normal(0, 0.1 * x_range)
            p0[4] += rng.normal(0, 0.1 * x_range)
            p0[2] *= np.exp(rng.normal(0, 0.3))
            p0[5] *= np.exp(rng.normal(0, 0.3))
            p0[0] *= np.exp(rng.normal(0, 0.3))
            p0[3] *= np.exp(rng.normal(0, 0.3))
            # Clip into bounds
            p0 = np.clip(p0, lower, upper)
        
        try:
            popt, _ = curve_fit(
                bimodal, x, y_pdf,
                p0=p0, bounds=(lower, upper),
                maxfev=20000
            )
            resid = np.sum((bimodal(x, *popt) - y_pdf) ** 2)
            if resid < best_resid:
                best_resid = resid
                best_popt = popt
        except RuntimeError:
            continue
    
    # If all fits failed, fall back to initial guess
    if best_popt is None:
        best_popt = np.array(p0_base)
    
    amp1, mu1, sig1, amp2, mu2, sig2 = best_popt
    
    # Convert amplitudes into normalized weights (fraction of total probability)
    # For a normalized Gaussian N(mu, sigma), area under amp * exp(...) is amp*sigma*sqrt(2pi)
    area1 = amp1 * sig1 * np.sqrt(2 * np.pi)
    area2 = amp2 * sig2 * np.sqrt(2 * np.pi)
    total = area1 + area2
    if total > 0:
        w1 = area1 / total
        w2 = area2 / total
    else:
        w1, w2 = 0.5, 0.5
    
    means = np.array([mu1, mu2])
    stds = np.array([sig1, sig2])
    weights = np.array([w1, w2])
    
    # Sort by mean (ascending) for consistent output
    order = np.argsort(means)
    means = means[order]
    stds = stds[order]
    weights = weights[order]
    
    # Generate smooth fitted PDF over the input range, normalized
    x_fit = np.linspace(x_min, x_max, len(x))
    y_fit = (
        weights[0] * np.exp(-0.5 * ((x_fit - means[0]) / stds[0]) ** 2)
        / (stds[0] * np.sqrt(2 * np.pi))
        + weights[1] * np.exp(-0.5 * ((x_fit - means[1]) / stds[1]) ** 2)
        / (stds[1] * np.sqrt(2 * np.pi))
    )
    
    return means, stds, weights, x_fit, y_fit

# def fit_gaussian(phi_deg_array, n_components=None, plot=False):
    
#     latitudes = []
#     standard_devs = []
#     weights = []

#     for _, phi_deg in enumerate(phi_deg_array):
#         bell = np.abs(phi_deg)
#         data_reshaped = bell.reshape(-1, 1)

#         if n_components is not None:
#             gmm = GaussianMixture(n_components=n_components, random_state=42)
#             gmm.fit(data_reshaped)
#             means = gmm.means_.flatten()
#             stds = np.sqrt(gmm.covariances_).flatten()
#             wgts = gmm.weights_
            
#             if plot:
#                 # Plotting
#                 x = np.linspace(bell.min(), bell.max(), 1000).reshape(-1, 1)
#                 logprob = gmm.score_samples(x)
#                 pdf = np.exp(logprob)
#                 plt.plot(x, pdf, label=f"{n_components}-Gaussian GMM", color="red")
        
#         else:
#             mean = np.mean(bell)
#             std = np.std(bell)
#             means = np.array([mean])
#             stds = np.array([std])
            
#             if plot: 
#                 # Plotting
#                 x = np.linspace(bell.min(), bell.max(), 1000)
#                 pdf = norm.pdf(x, loc=mean, scale=std)
#                 plt.plot(x, pdf, label="Single Gaussian", color="blue")

#         latitudes.append(means)
#         standard_devs.append(stds)
#         weights.append(wgts)

#         if plot:
#             # Histogram
#             plt.hist(bell, bins=50, density=True, alpha=0.5, label="Data")
#             plt.legend()
#             distribution_type = ["Unimodal", "Bimodal", "Trimodal"]
#             plt.title(f"{distribution_type[n_components - 1]} Gaussian Fit")
#             plt.xlabel("Value")
#             plt.ylabel("Density")
#             plt.ticklabel_format(style='plain', axis='x')

#             plt.show()
    
#             print("Means:", means)
#             print("Standard Deviations:", stds)
#             print("Weights:", weights)

#     return latitudes, standard_devs, weights

def fit_skew_normal(x, y):
    """
    Least-squares fit of a scaled skew-normal to (x, y), mirroring fit_gaussian.

        y ~ amp * 2 * phi((x-xi)/omega) * Phi(a * (x-xi)/omega) / omega

    Returns
    -------
    a_fit : float          shape (skewness)
    xi_fit : float         location
    omega_fit : float      scale
    amp_fit : float        amplitude (counts-per-unit)
    x_fit, y_fit : arrays  smooth curve on the input range
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    def skew_normal(x, amp, xi, omega, a):
        # standardized skew-normal shape, then scaled by amp
        z = (x - xi) / omega
        return amp * 2.0 * norm.pdf(z) * norm.cdf(a * z) / omega

    # ---- Initial guesses (mirroring fit_gaussian) ----
    mean_guess = np.sum(x * y) / np.sum(y)
    std_guess  = np.sqrt(np.sum(y * (x - mean_guess) ** 2) / np.sum(y))
    amp_guess  = np.max(y) * std_guess * np.sqrt(2 * np.pi)  # from peak ≈ amp/(omega√(2π))
    a_guess    = 0.0                                          # start symmetric

    # ---- Bounds: omega>0, amp>0, |a| bounded ----
    x_min, x_max = x.min(), x.max()
    span = x_max - x_min
    lower = [0.0,   x_min - span, 1e-6, -50.0]
    upper = [np.inf, x_max + span, span,  50.0]

    try:
        popt, _ = curve_fit(
            skew_normal, x, y,
            p0=[amp_guess, mean_guess, std_guess, a_guess],
            bounds=(lower, upper),
            maxfev=20000,
        )
        amp_fit, xi_fit, omega_fit, a_fit = popt
    except (RuntimeError, ValueError):
        amp_fit  = amp_guess
        xi_fit   = mean_guess
        omega_fit = std_guess
        a_fit    = a_guess

    # ---- Smooth curve over input range (same convention as fit_gaussian) ----
    x_fit = np.linspace(x_min, x_max, len(x))
    y_fit = skew_normal(x_fit, amp_fit, xi_fit, omega_fit, a_fit)

    return a_fit, xi_fit, omega_fit, amp_fit, x_fit, y_fit

def bootstrap_peak_periods(time, flux, fap_level, n_bootstraps=1000, boot_percent=0.8, 
                           min_period=5, max_period=20, n_freqs=10000, n_plot=100, figdir=None, sds=None): 
    """Bootstrap the peak periods from the Lomb-Scargle periodogram of the lightcurve.""" 
        # min, max period in days

    peak_periods = [] 
    n_data = len(time) 
    freq_grid = np.linspace(1/max_period, 1/min_period, n_freqs)
    np.random.seed(42)
    
    for i in tqdm(range(n_bootstraps)): 
        sample_indices = np.random.choice(n_data, size=int(boot_percent*n_data), replace=True) 
        sample_time = time[sample_indices] 
        sample_flux = flux[sample_indices] 
        ls = LombScargle(sample_time, sample_flux)
        power = ls.power(freq_grid)
        # Compute FAP only once at the start, as it should be ~the same for all LS
        if i == 0:
            fap = ls.false_alarm_level(fap_level) 
        else: 
            fap = fap

        peak_freqs, _ = get_peak_frequencies(freq_grid, power, [fap])
        
        if len(peak_freqs) == 0:
            continue  # no peaks found

        peak_periods.append(1 / np.array(peak_freqs))  

        if figdir and i in range(0, n_bootstraps, n_plot):
            os.makedirs(figdir, exist_ok=True)

            plt.plot(1/freq_grid, power, color='gray', alpha=0.5)
            plt.axhline(fap, color='red', linestyle='--', label=f'FAP={fap_level * 100}%' if i == 0 else None)
            plt.xlabel("Period [Days]")
            plt.ylabel("Power")
            plt.legend()
            plt.savefig(figdir + f"{sds}bootstrap.png", dpi=dpi)

    if len(peak_periods) == 0:
        return np.array([])
    
    return np.concatenate(peak_periods)


## POSTERIOR STATISTICAL CLASSIFICATION FUNCTIONS ##

# fit a gaussian to the mcmc posteriors

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


# def credible_interval(samples, ci=0.68):
#     """
#     Compute a credible interval for a (possibly skewed) distribution
#     using quantiles. Works for any posterior shape.

#     Parameters
#     ----------
#     samples : array-like
#         Posterior samples.
#     ci : float
#         Credible interval (e.g., 0.68, 0.90, 0.95).

#     Returns
#     -------
#     lower : float
#         Lower credible bound.
#     median : float
#         Median of the distribution.
#     upper : float
#         Upper credible bound.
#     """
#     samples = np.asarray(samples)

#     alpha = (1 - ci) / 2
#     lower = np.quantile(samples, alpha)
#     median = np.quantile(samples, 0.5)
#     upper = np.quantile(samples, 1 - alpha)

#     return lower, median, upper


def detect_truncation(samples, boundaries=[0, 90], delta_loglike=15):
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
    try:
        kde = gaussian_kde(samples)
    except np.linalg.LinAlgError:
        # If KDE fails due to singular matrix, treat as unimodal
        return "unimodal"
    xs = np.linspace(samples.min(), samples.max(), 2000)
    ys = kde(xs)

    # find all peaks
    peaks, _ = find_peaks(ys, prominence=min_prominence * np.max(ys))

    if len(peaks) < 2:
        return "unimodal"
    
    return "bimodal"


# def classify_posterior(samples, boundaries=[0, 90], min_prominence=0.01,
#                        allow_skew_truc=True, skew_threshold=1, verbose=True):  
#     samples = np.asarray(samples).ravel()

#     classification = detect_bimodality(samples, min_prominence=min_prominence)

#     if classification == "unimodal":
#         s = skew(samples)
#         mean, std = np.mean(samples), np.std(samples)
#         mu_hat_0, sigma_hat_0, _ = fit_truncated_normal(samples, mu0=mean, sigma0=std, b=boundaries[0])
#         mu_hat_1, sigma_hat_1, _ = fit_truncated_normal(samples, mu0=mean, sigma0=std, b=boundaries[1])

#         truc_bool, ll_low, ll_high = detect_truncation(samples, boundaries)
#         debug_print(verbose, f"Skewness: {s:.3f}, Mean: {mean:.3f}, Std: {std:.3f}, Mu0: {mu_hat_0:.3f}, Sigma0: {sigma_hat_0:.3f}, Mu1: {mu_hat_1:.3f}, Sigma1: {sigma_hat_1:.3f}")
#         print("skewness: ", s)

#         # Skewness dominates
#         if (abs(s) > skew_threshold) and allow_skew_truc: 
#             lower, median, upper = credible_interval(samples, ci=0.68)
#             lower_bound, upper_bound = np.abs(median - lower), np.abs(upper - median)
#             classification = "Skewed"
#             return classification, median, [lower_bound, upper_bound], None
        
#         if truc_bool and allow_skew_truc:
#             classification = "Truncated Gaussian"
#             # choose which boundary is better fit
#             if ll_low > ll_high:
#                 return classification, mu_hat_0, sigma_hat_0, None
#             else:
#                 return classification, mu_hat_1, sigma_hat_1, None

#         else: 
#             classification = "Gaussian"
#             lat, std, weights = fit_gaussian([samples], n_components=1, plot=False)
#             return classification, lat[0][0], std[0][0], weights

#     else:
#         classification = "Bimodal"
#         lat, std, weights = fit_gaussian([samples], n_components=2, plot=False)
#         return classification, lat[0], std[0], weights


def chi2_test(observed, expected, floor=1.0):
    observed = np.asarray(observed, dtype=float)
    expected = np.clip(np.asarray(expected, dtype=float), floor, None)
    return np.sum((observed - expected) ** 2 / expected)

def get_hist(samples):
    # Create histogram bins for observed data
    n_bins = int(np.sqrt(len(samples)))  # Sturges' rule alternative
    observed_freq, bin_edges = np.histogram(samples, bins=n_bins, density=False)
    observed_freq = np.array(observed_freq, dtype=float)
    zeromask = (observed_freq == 0)
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    observed_freq = np.interp(bin_centers, bin_centers[~zeromask], observed_freq[~zeromask])
    return n_bins, bin_centers, bin_edges, observed_freq

def classify_posterior(samples, truncbound=None, verbose=False):
    
    samples = np.asarray(samples).ravel()
    _, bin_centers, bin_edges, observed_freq = get_hist(samples)
    if (len(bin_centers) < 3):
        return 'gaussian', np.mean(bin_centers), 0, None, np.mean(bin_centers), observed_freq
    
    results = []

    # classification = detect_bimodality(samples, min_prominence=min_prominence)

    # if classification == "bimodal":
        
    # bimode
    bimeans, bistds, biweights, x_fit, biexpected_freq = fit_bimodal_gaussian(bin_centers, observed_freq)
    bimean, bistd, biwgt = bimeans, bistds, biweights

    biexpected_freq *= len(samples) * (bin_edges[1] - bin_edges[0])
    chi2_stat = chi2_test(observed_freq, biexpected_freq)
    results.append({
        'distribution': 'bimodal',
        'chi2_stat': chi2_stat,
        'params': {'mean': bimean, 'std': bistd},
        'fit': {'x': bin_centers, 'pdf': biexpected_freq}
    })

# else:
    mean, std = np.mean(samples), np.std(samples)

    # normal dist
    normmeans, normstds, x_fit, normexpected_freq = fit_gaussian(bin_centers, observed_freq)
    normmean, normstd = normmeans, normstds

    # Perform chi-squared goodness-of-fit tests
    
    # 1. Normal distribution
    # normexpected_freq = norm.pdf(bin_centers, normmean, normstd) * len(samples) * (bin_edges[1] - bin_edges[0])        
    chi2_stat = chi2_test(observed_freq, normexpected_freq)
    results.append({
        'distribution': 'normal',
        'chi2_stat': chi2_stat,
        'params': {'mean': normmean, 'std': normstd},
        'fit': {'x': bin_centers, 'pdf': normexpected_freq}
    })

    if truncbound:
        # truncated at lower bound
        mu_hat_0, sigma_hat_0, _ = fit_truncated_normal(samples, mu0=mean, sigma0=std, b=truncbound[0])

        # truncated at upper bound
        mu_hat_1, sigma_hat_1, _ = fit_truncated_normal(samples, mu0=mean, sigma0=std, b=truncbound[1]) 

        # 3. Truncated at lower bound
        a = (truncbound[0] - mu_hat_0) / sigma_hat_0
        b = np.inf
        trunc0expected_freq = truncnorm.pdf(bin_centers, a, b, mu_hat_0, sigma_hat_0) * len(samples) * (bin_edges[1] - bin_edges[0])
        chi2_stat = chi2_test(observed_freq, trunc0expected_freq)
        results.append({
            'distribution': 'truncated_at_0',
            'chi2_stat': chi2_stat,
            'params': {'mean': mu_hat_0, 'std': sigma_hat_0},
            'fit': {'x': bin_centers, 'pdf': trunc0expected_freq}
        })
        
        # 4. Truncated at upper bound
        a = -np.inf
        b = (truncbound[1] - mu_hat_1) / sigma_hat_1
        trunc1expected_freq = truncnorm.pdf(bin_centers, a, b, mu_hat_1, sigma_hat_1) * len(samples) * (bin_edges[1] - bin_edges[0])
        chi2_stat = chi2_test(observed_freq, trunc1expected_freq)
        results.append({
            'distribution': 'truncated_at_90',
            'chi2_stat': chi2_stat,
            'params': {'mean': mu_hat_1, 'std': sigma_hat_1},
            'fit': {'x': bin_centers, 'pdf': trunc1expected_freq}
        })
        
    # # 5. Skewed normal

    a_hat, xi_hat, omega_hat, amp_hat, _, skewexpected_freq = fit_skew_normal(bin_centers, observed_freq)

    lower  = skewnorm.ppf(0.16, a_hat, loc=xi_hat, scale=omega_hat)
    median_q = skewnorm.ppf(0.50, a_hat, loc=xi_hat, scale=omega_hat)
    upper  = skewnorm.ppf(0.84, a_hat, loc=xi_hat, scale=omega_hat)

    chi2_stat = chi2_test(observed_freq, skewexpected_freq)
    results.append({
        'distribution': 'skew_normal',
        'chi2_stat': chi2_stat,
        'params': {'mean': median_q, 'std': [median_q - lower, upper - median_q], 'alpha': a_hat},
        'fit': {'x': bin_centers, 'pdf': skewexpected_freq}
    })
    
    # Sort by chi2 statistic (lower is better fit)
    results.sort(key=lambda x: x['chi2_stat'])

    best_result = results[0]

    classification = best_result['distribution']
    if classification == 'bimodal':
        weights = biweights
    else: 
        weights = None 

    bestmean = best_result['params'].get('mean')
    beststd = best_result['params'].get('std')
    x = best_result['fit'].get('x')
    pdf = best_result['fit'].get('pdf')

    debug_print(verbose, [(d['distribution'], d['chi2_stat']) for d in results])

    return classification, bestmean, beststd, weights, x, pdf

def cluster_peaks(peaks, sds="sector", 
                  eps=0.005, 
                  min_samples=5, 
                  figdir=None, n_cols=2, 
                  n_bootstraps=10000, 
                  pass_frac=0.8, truncbound=None, verbose=False):

    debug_print(verbose, f"running dbscan with eps={eps}, min_samples={min_samples}")

    X = peaks.reshape(-1, 1)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_

    unique_labels = np.unique(labels)

    # --- Pre-compute which clusters will actually be plotted ---
    if figdir:
        valid_labels = []
        for label in unique_labels:
            count = len(peaks[labels == label].flatten())
            if count >= n_bootstraps * pass_frac:
                valid_labels.append(label)
        n_plots = len(valid_labels)
    else:
        n_plots = 0

    n_rows = int(np.ceil(n_plots / n_cols)) if n_plots > 0 else 1

    all_means, all_stds = [], []

    if figdir and n_plots > 0:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        axes = np.atleast_1d(axes).flatten()
    elif figdir:
        fig, axes = plt.subplots(1, 1, figsize=(5, 4))
        axes = np.array([axes])

    plot_idx = 0

    debug_print(verbose, "classifying clusters")
    for n, label in enumerate(unique_labels):
        debug_print(verbose, f"Processing cluster {n+1}/{len(unique_labels)} with label {label}")
        cluster_points = peaks[labels == label].flatten()
        count = len(cluster_points)

        if count < n_bootstraps * pass_frac:
            continue

        debug_print(verbose, f"Classifying Cluster {label}: count={count}, points={cluster_points}")
        classification, mean, std, weights, x, pdf = classify_posterior(cluster_points, 
                                                                    truncbound=truncbound, 
                                                                    verbose=verbose)
        
        if figdir:
            ax = axes[plot_idx]
            n_bins, bin_centers, _, observed_freq = get_hist(cluster_points)

            counts, bins, _ = ax.hist(cluster_points, bins=n_bins, color='lightsteelblue', edgecolor='k')
            ax.plot(x, pdf, color=plot_colors_rgb[1], label="Best Fit")
            ax.plot(bin_centers, observed_freq, color=plot_colors_rgb[3])
            ax.set_title(f"{label}, ct={count}, {classification}")
            ax.text(0.99, 0.99, f"mean={mean}", ha='right', va='top', transform=ax.transAxes)
            ax.text(0.99, 0.93, f"std={std}", ha='right', va='top', transform=ax.transAxes)
            ax.set_xlabel("Period [Days]")
            ax.set_ylabel("Counts")
            plot_idx += 1

        if classification == "bimodal":

            count1 = weights[0] * count
            count2 = weights[0] * count

            if (count1 < n_bootstraps * pass_frac) and (count2 < n_bootstraps * pass_frac): 
                continue
            elif (count1 >= n_bootstraps * pass_frac) and (count2 < n_bootstraps * pass_frac):
                mean = mean[0]
                std = std[0]
            elif (count1 < n_bootstraps * pass_frac) and (count2 >= n_bootstraps * pass_frac):
                mean = mean[1]
                std = std[1]
            else:
                mean = mean
                std = std

        all_means.append(mean)
        all_stds.append(std)

    if figdir:
        # Hide any unused subplots
        for j in range(plot_idx, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.savefig(figdir + f"{sds}cluster.png", dpi=dpi)

    return labels, all_means, all_stds


# def cluster_peaks(peaks, 
#                   eps=0.1, 
#                   min_samples=5, 
#                   allow_skew_truc=False, 
#                   plot=False, n_cols=2, 
#                   n_bootstraps=10000, 
#                   min_prominence=0.01,
#                   pass_frac=0.8, skew_threshold=1,
#                   verbose=False):
    
#     X = peaks.reshape(-1, 1)
#     db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
#     labels = db.labels_

#     unique_labels = np.unique(labels)
#     n_clusters = len(unique_labels)
#     n_rows = int(np.ceil(n_clusters / n_cols))

#     all_means, all_stds = [], []

#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
#     axes = axes.flatten()  

#     plot_idx = 0

#     for _, label in enumerate(unique_labels):
#         cluster_points = peaks[labels == label].flatten()
#         count = len(cluster_points)

#         if count < n_bootstraps * pass_frac:
#             continue

#         ax = axes[plot_idx]
#         counts, bins, _ = ax.hist(cluster_points, bins='auto', color='lightsteelblue', edgecolor='k')
#         if plot is False:
#             plt.close()  # closes the figure without displaying

#         classification, mean, std, weights = classify_posterior(cluster_points, 
#                                                        [np.min(cluster_points), 
#                                                         np.max(cluster_points)],
#                                                           allow_skew_truc=allow_skew_truc,
#                                                           skew_threshold=skew_threshold,
#                                                           min_prominence=min_prominence,
#                                                           verbose=verbose)
        
#         if plot:
#             ax.set_title(f"Cluster {label}: count={count}, classification={classification}")
#             ax.text(0.99, 0.99, f"mean={mean}", ha='right', va='top', transform=ax.transAxes)
#             ax.text(0.99, 0.93, f"std={std}", ha='right', va='top', transform=ax.transAxes)
#             ax.set_xlabel("Period [Days]")
#             ax.set_ylabel("Counts")
#             plot_idx += 1

#         if classification == "Bimodal":

#             count1 = weights[0][0] * count
#             count2 = weights[0][1] * count

#             if (count1 < n_bootstraps * pass_frac) and (count2 < n_bootstraps * pass_frac): 
#                 continue
#             elif (count1 >= n_bootstraps * pass_frac) and (count2 < n_bootstraps * pass_frac):
#                 mean = mean[0]
#                 std = std[0]
#             elif (count1 < n_bootstraps * pass_frac) and (count2 >= n_bootstraps * pass_frac):
#                 mean = mean[1]
#                 std = std[1]
#             else:
#                 mean = mean
#                 std = std

#         # if gaussian:
#         #     if len(cluster_points) < 2:
#         #         mean = np.mean(cluster_points)
#         #         std = np.std(cluster_points)
#         #     else:
#         #         # print(cluster_points.shape)
#         #         mean, std = fit_gaussian([cluster_points], n_components=1, plot=False)
#         #         x = np.linspace(bins[0], bins[-1], 1000)
#         #         y = norm.pdf(x, loc=mean, scale=std)
#         #         y_scaled = y * (counts.max() / y.max())

#         #         if plot:
#         #             ax.plot(x, y_scaled[0, :], 'r-', linewidth=2, label='Gaussian fit')

#         #         mean = mean[0][0]
#         #         std = std[0][0]
#         # else:
#         #     mean = np.mean(cluster_points)
#         #     std = np.std(cluster_points)

#         all_means.append(mean)
#         all_stds.append(std)

#     if plot:
#         # Hide any unused subplots
#         for j in range(plot_idx, len(axes)):
#             fig.delaxes(axes[j])

#         plt.tight_layout()
#         plt.show()

#     return labels, all_means, all_stds


def save_bootstrap(sector_data_list, sector_data_strings, save_dir, flux_type='detrended', fap_level=0.01, min_period_arr=[], max_period_arr=[],
                   n_freqs=int(1e5), n_bootstraps=10000, figdir=None):
    
    os.makedirs(save_dir, exist_ok=True)

    for i, sector_data in enumerate(sector_data_list):

        peak_periods = bootstrap_peak_periods(sector_data['time'], sector_data[flux_type], fap_level, 
                                                min_period=min_period_arr[i], max_period=max_period_arr[i], n_freqs=n_freqs, 
                                                n_bootstraps=n_bootstraps, figdir=figdir, sds=sector_data_strings[i])
        # labels, all_means, all_stds = cluster_peaks(peak_periods, eps=0.0001, plot=True, n_cols=3)
        np.savez(save_dir + f'{sector_data_strings[i]}_bootstrap.npz', 
                 peak_periods=peak_periods)#, labels=labels, all_means=all_means, all_stds=all_stds)


def flatten_mixed_list(mixed_list):
    result = []
    for item in mixed_list:
        if isinstance(item, np.ndarray):
            result.extend(item)
        else:
            result.append(item)
    return result


# def save_cluster(periodograms, 
#                  peak_periods_list, 
#                  sector_data_strings, 
#                  save_dir, 
#                  plot=False, 
#                  eps_arr=[0.001, 0.001, 0.001, 0.001, 0.005], 
#                  tolerance= 0.005, 
#                  n_cols=3, 
#                  n_bootstraps=10000, 
#                  min_prominence_arr=[0.5, 0.5, 0.5, 0.5, 0.75],
#                  pass_frac=0.8,
#                  verbose=False):
    
#     for i, peak_periods in enumerate(peak_periods_list):
#         _, all_means, all_stds = cluster_peaks(peak_periods, 
#                                                eps=eps_arr[i], 
#                                                plot=plot, 
#                                                n_cols=n_cols, 
#                                                n_bootstraps=n_bootstraps,
#                                                min_prominence=min_prominence_arr[i],
#                                                pass_frac=pass_frac,
#                                                verbose=verbose)

#         means_flat = flatten_mixed_list(all_means)
#         stds_flat = flatten_mixed_list(all_stds)

#         peak_periodogram_periods = np.array(1/periodograms[i]['peak_freqs'])

#         # print(all_means, all_stds)
#         xmatch = np.abs(peak_periodogram_periods[:, np.newaxis] - np.array(means_flat))
#         potential_matches = np.abs(xmatch) < tolerance
#         closest_matches, _ = np.unique(np.where(potential_matches)[1], return_counts=True)

#         matched_means = np.array(means_flat)[closest_matches]
#         matched_stds = np.array(stds_flat)[closest_matches]

#         print("Sector:", sector_data_strings[i])
#         print(f"{len(closest_matches)} out of {len(means_flat)} All means:", means_flat)
#         print(f"{len(peak_periodogram_periods)} Peak periods from periodogram:", peak_periodogram_periods)
        
#         print(f"matched means:", matched_means)
#         print(f"matched stds:", matched_stds)
#         print()
#         np.savez(save_dir + f'{sector_data_strings[i]}_clustered_peaks.npz', 
#                  matched_means=matched_means, matched_stds=matched_stds)


def save_cluster(periodograms, 
                 peak_periods_list, 
                 sector_data_strings, 
                 save_dir, 
                 figdir=None, 
                 eps_arr=[0.001, 0.001, 0.001, 0.001, 0.005], 
                 tolerance= 0.005, 
                 n_cols=3, 
                 n_bootstraps=10000, 
                 pass_frac=0.8,
                 truncbound=None, 
                 verbose=False):

    os.makedirs(save_dir, exist_ok=True)

    for i, peak_periods in enumerate(peak_periods_list):

        _, all_means, all_stds = cluster_peaks(peak_periods, 
                                                sds=sector_data_strings[i],
                                                eps=eps_arr[i], 
                                                figdir=figdir, n_cols=n_cols, 
                                                n_bootstraps=n_bootstraps, 
                                                pass_frac=pass_frac, 
                                                truncbound=truncbound, 
                                                verbose=verbose)
        means_flat = flatten_mixed_list(all_means)
        stds_flat = flatten_mixed_list(all_stds)

        peak_periodogram_periods = np.array(1/periodograms[i]['peak_freqs'])

        # print(all_means, all_stds)
        xmatch = np.abs(peak_periodogram_periods[:, np.newaxis] - np.array(means_flat))
        potential_matches = np.abs(xmatch) < tolerance
        closest_matches, _ = np.unique(np.where(potential_matches)[1], return_counts=True)

        matched_means = np.array(means_flat)[closest_matches]
        matched_stds = [stds_flat[i] for i in closest_matches]

        print("Sector:", sector_data_strings[i])
        print(f"{len(closest_matches)} out of {len(means_flat)} All means:", means_flat)
        print(f"{len(peak_periodogram_periods)} Peak periods from periodogram:", peak_periodogram_periods)
        
        print(f"matched means:", matched_means)
        print(f"matched stds:", matched_stds)
        print()
        np.savez(save_dir + f'{sector_data_strings[i]}_clustered_peaks.npz', 
                 matched_means=matched_means, matched_stds=np.array(matched_stds, dtype=object))


