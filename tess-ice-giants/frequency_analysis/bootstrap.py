import warnings

import numpy as np
import matplotlib.pyplot as plt

from astropy.timeseries import LombScargle
from scipy import optimize, stats
from scipy.signal import find_peaks
from scipy.stats import norm, skew, gaussian_kde
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

from fullsector import debug_print, get_peak_frequencies

warnings.filterwarnings('ignore') 

def fit_gaussian(phi_deg_array, n_components=None, plot=False):
    
    latitudes = []
    standard_devs = []
    weights = []

    for _, phi_deg in enumerate(phi_deg_array):
        bell = np.abs(phi_deg)
        data_reshaped = bell.reshape(-1, 1)

        if n_components is not None:
            gmm = GaussianMixture(n_components=n_components, random_state=42)
            gmm.fit(data_reshaped)
            means = gmm.means_.flatten()
            stds = np.sqrt(gmm.covariances_).flatten()
            wgts = gmm.weights_
            
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
        weights.append(wgts)

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
            print("Weights:", weights)

    return latitudes, standard_devs, weights


def bootstrap_peak_periods(time, flux, fap_level, n_bootstraps=1000, boot_percent=0.7, 
                           min_period=5, max_period=20, n_freqs=10000, plot=False, n_plot=100): 
    """Bootstrap the peak periods from the Lomb-Scargle periodogram of the lightcurve.""" 
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

        if plot and i in range(0, n_bootstraps, n_plot):
            plt.plot(1/freq_grid, power, color='gray', alpha=0.5)
            plt.axhline(fap, color='red', linestyle='--', label=f'FAP={fap_level * 100}%' if i == 0 else None)
            plt.xlabel("Period [Days]")
            plt.ylabel("Power")
    
    if plot: 
        plt.legend()
        plt.show()
    
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


def classify_posterior(samples, boundaries=[0, 90], min_prominence=0.01,
                       allow_skew_truc=True, verbose=True):  
    samples = np.asarray(samples).ravel()

    classification = detect_bimodality(samples, min_prominence=min_prominence)

    if classification == "unimodal":
        s = skew(samples)
        mean, std = np.mean(samples), np.std(samples)
        mu_hat_0, sigma_hat_0, _ = fit_truncated_normal(samples, mu0=mean, sigma0=std, b=boundaries[0])
        mu_hat_1, sigma_hat_1, _ = fit_truncated_normal(samples, mu0=mean, sigma0=std, b=boundaries[1])

        truc_bool, ll_low, ll_high = detect_truncation(samples, boundaries)
        debug_print(verbose, f"Skewness: {s:.3f}, Mean: {mean:.3f}, Std: {std:.3f}, Mu0: {mu_hat_0:.3f}, Sigma0: {sigma_hat_0:.3f}, Mu1: {mu_hat_1:.3f}, Sigma1: {sigma_hat_1:.3f}")
        
        # Skewness dominates
        if (abs(s) > 1) and allow_skew_truc: 
            lower, median, upper = credible_interval(samples, ci=0.68)
            lower_bound, upper_bound = np.abs(median - lower), np.abs(upper - median)
            classification = "Skewed"
            return classification, median, [lower_bound, upper_bound], None
        
        if truc_bool and allow_skew_truc:
            classification = "Truncated Gaussian"
            # choose which boundary is better fit
            if ll_low > ll_high:
                return classification, mu_hat_0, sigma_hat_0, None
            else:
                return classification, mu_hat_1, sigma_hat_1, None

        else: 
            classification = "Gaussian"
            lat, std, weights = fit_gaussian([samples], n_components=1, plot=False)
            return classification, lat[0][0], std[0][0], weights

    else:
        classification = "Bimodal"
        lat, std, weights = fit_gaussian([samples], n_components=2, plot=False)
        return classification, lat[0], std[0], weights

def cluster_peaks(peaks, 
                  eps=0.1, 
                  min_samples=5, 
                  allow_skew_truc=False, 
                  plot=False, n_cols=2, 
                  n_bootstraps=10000, 
                  min_prominence=0.01,
                  pass_frac=0.8,
                  verbose=False):
    
    X = peaks.reshape(-1, 1)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    n_rows = int(np.ceil(n_clusters / n_cols))

    all_means, all_stds = [], []

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten()  

    plot_idx = 0

    for _, label in enumerate(unique_labels):
        cluster_points = peaks[labels == label].flatten()
        count = len(cluster_points)

        if count < n_bootstraps * pass_frac:
            continue

        ax = axes[plot_idx]
        counts, bins, _ = ax.hist(cluster_points, bins='auto', color='lightsteelblue', edgecolor='k')
        if plot is False:
            plt.close()  # closes the figure without displaying

        classification, mean, std, weights = classify_posterior(cluster_points, 
                                                       [np.min(cluster_points), 
                                                        np.max(cluster_points)],
                                                          allow_skew_truc=allow_skew_truc,
                                                          min_prominence=min_prominence,
                                                          verbose=verbose)
        
        if plot:
            ax.set_title(f"Cluster {label}: count={count}, classification={classification}")
            ax.text(0.99, 0.99, f"mean={mean}", ha='right', va='top', transform=ax.transAxes)
            ax.text(0.99, 0.93, f"std={std}", ha='right', va='top', transform=ax.transAxes)
            ax.set_xlabel("Period [Days]")
            ax.set_ylabel("Counts")
            plot_idx += 1

        if classification == "Bimodal":

            count1 = weights[0][0] * count
            count2 = weights[0][1] * count

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

        # if gaussian:
        #     if len(cluster_points) < 2:
        #         mean = np.mean(cluster_points)
        #         std = np.std(cluster_points)
        #     else:
        #         # print(cluster_points.shape)
        #         mean, std = fit_gaussian([cluster_points], n_components=1, plot=False)
        #         x = np.linspace(bins[0], bins[-1], 1000)
        #         y = norm.pdf(x, loc=mean, scale=std)
        #         y_scaled = y * (counts.max() / y.max())

        #         if plot:
        #             ax.plot(x, y_scaled[0, :], 'r-', linewidth=2, label='Gaussian fit')

        #         mean = mean[0][0]
        #         std = std[0][0]
        # else:
        #     mean = np.mean(cluster_points)
        #     std = np.std(cluster_points)

        all_means.append(mean)
        all_stds.append(std)

    if plot:
        # Hide any unused subplots
        for j in range(plot_idx, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    return labels, all_means, all_stds

def save_bootstrap(sector_data_list, sector_data_strings, root, flux_type='detrended', fap_level=0.1, min_period_arr=[], max_period_arr=[],
                   n_freqs=int(1e5), n_bootstraps=10000, plot=True):
    
    for i, sector_data in enumerate(sector_data_list):

        peak_periods = bootstrap_peak_periods(sector_data['time'], sector_data[flux_type], fap_level, 
                                                min_period=min_period_arr[i], max_period=max_period_arr[i], n_freqs=n_freqs, 
                                                n_bootstraps=n_bootstraps, plot=plot)
        # labels, all_means, all_stds = cluster_peaks(peak_periods, eps=0.0001, plot=True, n_cols=3)
        np.savez(root + f'{sector_data_strings[i]}_bootstrap.npz', 
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
                 plot=False, 
                 eps_arr=[0.001, 0.001, 0.001, 0.001, 0.005], 
                 tolerance= 0.005, 
                 n_cols=3, 
                 n_bootstraps=10000, 
                 min_prominence_arr=[0.5, 0.5, 0.5, 0.5, 0.75],
                 pass_frac=0.8,
                 verbose=False):
    
    for i, peak_periods in enumerate(peak_periods_list):
        _, all_means, all_stds = cluster_peaks(peak_periods, 
                                               eps=eps_arr[i], 
                                               plot=plot, 
                                               n_cols=n_cols, 
                                               n_bootstraps=n_bootstraps,
                                               min_prominence=min_prominence_arr[i],
                                               pass_frac=pass_frac,
                                               verbose=verbose)

        means_flat = flatten_mixed_list(all_means)
        stds_flat = flatten_mixed_list(all_stds)

        peak_periodogram_periods = np.array(1/periodograms[i]['peak_freqs'])

        # print(all_means, all_stds)
        xmatch = np.abs(peak_periodogram_periods[:, np.newaxis] - np.array(means_flat))
        potential_matches = np.abs(xmatch) < tolerance
        closest_matches, _ = np.unique(np.where(potential_matches)[1], return_counts=True)

        matched_means = np.array(means_flat)[closest_matches]
        matched_stds = np.array(stds_flat)[closest_matches]

        print("Sector:", sector_data_strings[i])
        print(f"{len(closest_matches)} out of {len(means_flat)} All means:", means_flat)
        print(f"{len(peak_periodogram_periods)} Peak periods from periodogram:", peak_periodogram_periods)
        
        print(f"matched means:", matched_means)
        print(f"matched stds:", matched_stds)
        print()
        np.savez(save_dir + f'{sector_data_strings[i]}_clustered_peaks.npz', 
                 matched_means=matched_means, matched_stds=matched_stds)

