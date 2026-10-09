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
    
    # from scipy.signal import find_peaks

    peaks, _ = find_peaks(y, height=y.max() * 0.05)
    if len(peaks) >= 2:
        top2 = peaks[np.argsort(y[peaks])][::-1][:2]
        top2 = np.sort(top2)
        mu1_g, mu2_g = x[top2[0]], x[top2[1]]
    else:
        mu1_g, mu2_g = x.min() + 0.25 * (x.max() - x.min()), x.min() + 0.75 * (x.max() - x.min())

    sig1_g = max(dx, (x.max() - x.min()) / 10)
    sig2_g = sig1_g
    amp1_g = max(amp_guess, 1e-6)
    amp2_g = max(amp_guess * 0.5, 1e-6)
    p0_base = [amp1_g, mu1_g, sig1_g, amp2_g, mu2_g, sig2_g]
    
    # Bounds: amplitudes >= 0, stds > 0
    x_min, x_max = x.min(), x.max()
    x_range = x_max - x_min
    lower = [0,          x_min, 1e-6, 0,          x_min, 1e-6]
    upper = [np.inf,     x_max, x_range, np.inf, x_max, x_range]
    
    best_popt = None
    best_resid = np.inf
    
    rng = np.random.default_rng(42)
    lower = np.array(lower, dtype=float)
    upper = np.array(upper, dtype=float)

    for i in range(n_restarts):
        p0 = np.array(p0_base, dtype=float)
        if i > 0:
            p0[1] += rng.normal(0, 0.1 * x_range)
            p0[4] += rng.normal(0, 0.1 * x_range)
            p0[2] *= np.exp(rng.normal(0, 0.3))
            p0[5] *= np.exp(rng.normal(0, 0.3))
            p0[0] *= np.exp(rng.normal(0, 0.3))
            p0[3] *= np.exp(rng.normal(0, 0.3))
        
        # Always clip, including i==0, with a small margin inside the bounds
        p0 = np.clip(p0, lower + 1e-9, upper - 1e-9)
        
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
        except (RuntimeError, ValueError):
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
    if figdir:
        fig, ax = plt.subplots(figsize=(10, 6))   # <-- new figure

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

            ax.plot(1/freq_grid, power, color='gray', alpha=0.5)
            ax.axhline(fap, color='red', linestyle='--',
                    label=f'FAP={fap_level * 100}%' if i == 0 else None)
            ax.set_xlabel("Period [Days]")
            ax.set_ylabel("Power")
            ax.legend()
        
    if figdir:
        fig.savefig(os.path.join(figdir, f"{sds}bootstrap.png"), dpi=dpi)

    if len(peak_periods) == 0:
        return np.array([])
    
    return np.concatenate(peak_periods)


## POSTERIOR STATISTICAL CLASSIFICATION FUNCTIONS ##

def fit_truncated_normal(x, bound, direction, mu0=None, sigma0=None):

    x = np.asarray(x, dtype=float)
    if mu0 is None:    mu0    = np.mean(x)
    if sigma0 is None: sigma0 = np.std(x, ddof=1)

    if direction == "lower":
        a, b = (bound - mu0) / sigma0, np.inf   # placeholder; recomputed per-iter
    elif direction == "upper":
        a, b = -np.inf, (bound - mu0) / sigma0
    else:
        raise ValueError("direction must be 'lower' or 'upper'")

    def neg_loglike(params):
        mu, log_sigma = params
        sigma = np.exp(log_sigma)
        z = (x - mu) / sigma
        log_pdf = stats.norm.logpdf(z) - np.log(sigma)
        if direction == "lower":
            # P(X >= bound) = sf((bound - mu)/sigma)
            log_norm = np.log(stats.norm.sf((bound - mu) / sigma))
        else:
            # P(X <= bound) = cdf((bound - mu)/sigma)
            log_norm = np.log(stats.norm.cdf((bound - mu) / sigma))
        return -np.sum(log_pdf - log_norm)

    res = optimize.minimize(
        neg_loglike,
        x0=[mu0, np.log(sigma0)],
        method="L-BFGS-B",
    )
    mu_hat    = res.x[0]
    sigma_hat = np.exp(res.x[1])

    # Build the matching truncated-normal PDF on a dense grid
    if direction == "lower":
        a_std, b_std = (bound - mu_hat) / sigma_hat, np.inf
    else:
        a_std, b_std = -np.inf, (bound - mu_hat) / sigma_hat

    x_fit = np.linspace(x.min(), x.max(), 200)
    y_fit_pdf = stats.truncnorm.pdf(x_fit, a_std, b_std, loc=mu_hat, scale=sigma_hat)

    return mu_hat, sigma_hat, x_fit, y_fit_pdf

def chi2_test(observed, expected, dof, floor_frac=0.01):
    observed = np.asarray(observed, dtype=float)
    expected = np.asarray(expected, dtype=float)
    floor = max(floor_frac * expected.max(), 1e-6)
    expected = np.clip(expected, floor, None)
    return np.sum((observed - expected) ** 2 / expected) / dof

def get_hist(samples, n_bins=None, mask=True):
    # Create histogram bins for observed data
    if n_bins is None:
        n_bins = int(np.sqrt(len(samples)))  # Sturges' rule alternative
    observed_freq, bin_edges = np.histogram(samples, bins=n_bins, density=False)
    observed_freq = np.array(observed_freq, dtype=float)
    zeromask = (observed_freq == 0)
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    if mask == True:
        observed_freq = np.interp(bin_centers, bin_centers[~zeromask], observed_freq[~zeromask])
    
    return n_bins, bin_centers, bin_edges, observed_freq

def poisson_deviance(observed, expected):
    # avoid log(0)
    eps = 1e-12
    obs = np.asarray(observed, dtype=float)
    exp = np.asarray(expected, dtype=float) + eps
    term = np.where(obs > 0, obs * np.log(obs / exp), 0.0)
    return 2.0 * np.sum(term - (obs - exp))

def classify_posterior(samples, truncbound=None, 
                       n_bins=None,
                       floor_frac=0.01, 
                       promfrac=None,
                       mask=True,
                       verbose=False):

    print("mask?", mask)
    
    samples = np.asarray(samples).ravel()
    _, bin_centers, bin_edges, observed_freq = get_hist(samples, 
                                                        n_bins=n_bins, 
                                                        mask=mask)

    print("Observed Freq Array:")
    print(observed_freq)

    # nonzero_freqs = observed_freq[observed_freq != 0]
    # print(f'there are {len(nonzero_freqs)} bins w freq >= 1')
    debug_print(verbose, f"len unique {len(np.unique(observed_freq))}")
    if (len(np.unique(observed_freq)) < 2):
        return 'normal', np.mean(bin_centers), 0, None, bin_centers, observed_freq
    
    dx  = bin_edges[1] - bin_edges[0]
    N   = len(samples)
    nb  = len(bin_centers)
    # log_nb = np.log(nb)

    results = []

    def add_result(name, chi_2, k, params, freq):
        results.append({
            'distribution': name,
            'chi2': chi_2,          # keep old key for downstream compatibility
            'k': k,
            'params': params,
            'fit': {'x': bin_centers, 'pdf': freq},
        })

    # classification = detect_bimodality(samples, min_prominence=min_prominence)

    # if classification == "bimodal":
        
    # bimode
    bimeans, bistds, biweights, x_fit, biexpected_freq = fit_bimodal_gaussian(bin_centers, observed_freq)

    biexpected_freq *= N * dx
    bichi2_stat = chi2_test(observed_freq, biexpected_freq, dof=6, floor_frac=floor_frac)
    # bichi2_stat = (0, bichi2_stat[1])
    sep = abs(bimeans[1] - bimeans[0])
    width = np.max(bistds)
    if (sep > width) and (promfrac is None):   # tune this

        add_result('bimodal', bichi2_stat, 6, 
                {'mean': bimeans, 'std':bistds, 'weights':biweights},
                biexpected_freq)

    if promfrac:
    # Pad on both sides so peaks at the first/last bin are detected


        padded = np.concatenate(([0], observed_freq, [0]))

        peaks_padded, props = find_peaks(
            padded, prominence=np.max(observed_freq) * promfrac
        )

        # Shift indices back to original (unpadded) array coordinates
        peaks = peaks_padded #- 1

        debug_print(verbose, f"{len(peaks)} peaks")

        if len(peaks) > 1:
            return 'bimodal', bimeans, bistds, biweights, bin_centers, biexpected_freq

# else:
    mean, std = np.mean(samples), np.std(samples)

    # normal dist
    normmean, normstd, x_fit, normexpected_freq = fit_gaussian(bin_centers, observed_freq)

    # 1. Normal distribution
    # normexpected_freq = norm.pdf(bin_centers, normmean, normstd) * N * dx        
    normchi2_stat = chi2_test(observed_freq, normexpected_freq, dof=2, floor_frac=floor_frac)
    add_result('normal', normchi2_stat, 2,
               {'mean': normmean, 'std': normstd}, normexpected_freq)

    if truncbound:
        # truncated at lower bound
        mu_hat_0, sigma_hat_0, _, _ = fit_truncated_normal(samples, bound=truncbound[0], direction='lower', mu0=mean, sigma0=std)

        # truncated at upper bound
        mu_hat_1, sigma_hat_1, _, _ = fit_truncated_normal(samples, bound=truncbound[1], direction='upper', mu0=mean, sigma0=std) 

        # 3. Truncated at lower bound
        a = (truncbound[0] - mu_hat_0) / sigma_hat_0
        b = np.inf
        trunc0expected_freq = truncnorm.pdf(bin_centers, a, b, mu_hat_0, sigma_hat_0) * N * dx
        trunc0chi2_stat = chi2_test(observed_freq, trunc0expected_freq, dof=2, floor_frac=floor_frac)

        add_result('truncated_at_0', trunc0chi2_stat, 2, 
                   {'mean': mu_hat_0, 'std': sigma_hat_0}, trunc0expected_freq)
        
        # 4. Truncated at upper bound
        a = -np.inf
        b = (truncbound[1] - mu_hat_1) / sigma_hat_1
        trunc1expected_freq = truncnorm.pdf(bin_centers, a, b, mu_hat_1, sigma_hat_1) * N * dx
        trunc1chi2_stat = chi2_test(observed_freq, trunc1expected_freq, dof=2, floor_frac=floor_frac)

        add_result('truncated_at_1', trunc1chi2_stat, 2, 
                   {'mean': mu_hat_1, 'std': sigma_hat_1}, trunc1expected_freq)
        
    # # 5. Skewed normal

    a_hat, xi_hat, omega_hat, amp_hat, _, skewexpected_freq = fit_skew_normal(bin_centers, observed_freq)

    lower  = skewnorm.ppf(0.16, a_hat, loc=xi_hat, scale=omega_hat)
    median_q = skewnorm.ppf(0.50, a_hat, loc=xi_hat, scale=omega_hat)
    upper  = skewnorm.ppf(0.84, a_hat, loc=xi_hat, scale=omega_hat)

    skewchi2_stat = chi2_test(observed_freq, skewexpected_freq, dof=3, floor_frac=floor_frac)
    add_result('skew_normal', skewchi2_stat, 3,
               {'mean': median_q,
                'std': [median_q - lower, upper - median_q],
                'alpha': a_hat},
               skewexpected_freq)
    
    # Sort by chi2 statistic (lower is better fit)
    # results.sort(key=lambda x: x['chi2'])

    for r in results:
        #r['bic'] = r['chi2'] + r['k'] * np.log(n_bins)
        r['bic'] = poisson_deviance(observed_freq, r['fit']['pdf']) + r['k'] * np.log(n_bins)

    results.sort(key=lambda x: x['bic'])

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

    debug_print(verbose, [(d['distribution'], d['bic']) for d in results])

    is_scalar_mean = np.ndim(bestmean) == 0

    if truncbound and is_scalar_mean and (bestmean > truncbound[1]):
        bestmean = truncbound[1]
    if truncbound and is_scalar_mean and (bestmean < truncbound[0]):
        bestmean = truncbound[0]    
    return classification, bestmean, beststd, weights, x, pdf

def cluster_peaks(peaks, sds="sector", 
                  eps=0.005, 
                  min_samples=5, 
                  figdir=None, n_cols=2, 
                  n_bootstraps=10000, 
                  pass_frac=0.8, 
                  truncbound=None, 
                  floor_frac=0.01, 
                  n_bins=None,
                  verbose=False):

    os.makedirs(figdir, exist_ok=True)
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
                                                                    n_bins=n_bins, 
                                                                    floor_frac=floor_frac, 
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

        if len(all_means) > 0:
            plt.savefig(figdir + f"{sds}cluster.png", dpi=dpi)

    return labels, all_means, all_stds

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

