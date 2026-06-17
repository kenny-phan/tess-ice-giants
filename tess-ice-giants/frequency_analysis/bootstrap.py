import numpy as np
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from astropy.timeseries import LombScargle
from scipy.stats import norm
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from fullsector import get_peak_frequencies

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


def cluster_peaks(peaks, eps=0.1, min_samples=5, gaussian=True, plot=False, n_cols=2, n_bootstraps=10000):
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

        if count < n_bootstraps * 0.8:
            continue

        ax = axes[plot_idx]
        counts, bins, _ = ax.hist(cluster_points, bins='auto', color='lightsteelblue', edgecolor='k')
        if plot is False:
            plt.close()  # closes the figure without displaying

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

def save_bootstrap(sector_data_list, sector_data_strings, root, flux_type='detrended', fap_level=0.1, min_period_arr=[], max_period_arr=[],
                   n_freqs=int(1e5), n_bootstraps=10000, plot=True):
    
    for i, sector_data in enumerate(sector_data_list):

        peak_periods = bootstrap_peak_periods(sector_data['time'], sector_data[flux_type], fap_level, 
                                                min_period=min_period_arr[i], max_period=max_period_arr[i], n_freqs=n_freqs, 
                                                n_bootstraps=n_bootstraps, plot=plot)
        # labels, all_means, all_stds = cluster_peaks(peak_periods, eps=0.0001, plot=True, n_cols=3)
        np.savez(root + f'{sector_data_strings[i]}_bootstrap.npz', 
                 peak_periods=peak_periods)#, labels=labels, all_means=all_means, all_stds=all_stds)
        

def save_cluster(periodograms, peak_periods_list, sector_data_strings, save_dir, 
                 plot=False, eps=0.0001, tolerance= 0.005, ncols=3, n_bootstraps=10000):
    for i, peak_periods in enumerate(peak_periods_list):
        _, all_means, all_stds = cluster_peaks(peak_periods, eps=eps, plot=plot, n_cols=ncols, n_bootstraps=n_bootstraps)
        peak_periodogram_periods = np.array(1/periodograms[i]['peaks'])
        xmatch = np.abs(peak_periodogram_periods[:, np.newaxis] - np.array(all_means))
        potential_matches = np.abs(xmatch) < tolerance
        closest_matches, _ = np.unique(np.where(potential_matches)[1], return_counts=True)

        matched_means = np.array(all_means)[closest_matches]
        matched_stds = np.array(all_stds)[closest_matches]

        print("Sector:", sector_data_strings[i])
        print(f"{len(closest_matches)} out of {len(all_means)} All means:", all_means)
        print(f"{len(peak_periodogram_periods)} Peak periods from periodogram:", peak_periodogram_periods)
        
        print(f"matched means:", matched_means)
        print(f"matched stds:", matched_stds)
        print()
        np.savez(save_dir + f'{sector_data_strings[i]}_clustered_peaks.npz', 
                 matched_means=matched_means, matched_stds=matched_stds)