import numpy as np
import matplotlib.pyplot as plt

from astropy.timeseries import LombScargle
from scipy.stats import norm
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from frequency_analysis.frequency_processing import get_peak_frequencies
from frequency_analysis.mcmc import fit_gaussian

def bootstrap_peak_periods(time, flux, fap_level=0.01, n_bootstraps=1000, boot_percent=0.7, 
                           min_period=5, max_period=20, n_freqs=1000, plot=False, n_plot=100): 
    """Bootstrap the peak periods from the Lomb-Scargle periodogram of the lightcurve.""" 
    peak_periods = [] 
    n_data = len(time) 
    freq_grid = np.linspace(1/max_period, 1/min_period, n_freqs)
    
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
    
    plt.legend()
    
    return np.concatenate(peak_periods)


def cluster_peaks(peaks, eps=0.1, min_samples=5, gaussian=True, plot=False, n_cols=2):
    X = peaks.reshape(-1, 1)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    n_rows = int(np.ceil(n_clusters / n_cols))

    all_means, all_stds = [], []

    if plot: 
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        axes = axes.flatten()  

        plot_idx = 0

    for _, label in enumerate(unique_labels):
        cluster_points = peaks[labels == label].flatten()
        count = len(cluster_points)

        if count < len(peaks) / 100:
            continue

        if plot:
            ax = axes[plot_idx]
            counts, bins, _ = ax.hist(cluster_points, bins='auto', color='lightsteelblue', edgecolor='k')

        if gaussian:
            if len(cluster_points) < 2:
                mean = np.mean(cluster_points)
                std = np.std(cluster_points)
            else:
                # print(cluster_points.shape)
                mean, std = fit_gaussian([cluster_points], n_components=1, plot=False)
                x = np.linspace(bins[0], bins[-1], 1000)
                y = norm.pdf(x, loc=mean, scale=std)
                y_scaled = y * (counts.max() / y.max())

                if plot:
                    ax.plot(x, y_scaled[0, :], 'r-', linewidth=2, label='Gaussian fit')

                mean = mean[0][0]
                std = std[0][0]
        else:
            mean = np.mean(cluster_points)
            std = np.std(cluster_points)

        all_means.append(mean)
        all_stds.append(std)

        if plot:
            ax.set_title(f"Cluster {label}: count={count}")
            ax.text(0.99, 0.99, f"mean={mean:.3f}", ha='right', va='top', transform=ax.transAxes)
            ax.text(0.99, 0.93, f"std={std:.3f}", ha='right', va='top', transform=ax.transAxes)
            ax.set_xlabel("Period [Days]")
            ax.set_ylabel("Counts")
            plot_idx += 1

    if plot:
        # Hide any unused subplots
        for j in range(plot_idx, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    return labels, all_means, all_stds

def z_score(data):
    return (data - np.nanmedian(data)) / np.nanstd(data)

def chi2(O, E):
    return np.nansum((O - E)**2 / E)