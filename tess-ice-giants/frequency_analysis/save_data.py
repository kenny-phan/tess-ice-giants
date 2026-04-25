import numpy as np
from tqdm import tqdm

from frequency_analysis.frequency_processing import *
from frequency_analysis.long_term_trends import bootstrap_peak_periods, cluster_peaks
from frequency_analysis.mcmc import *
from frequency_analysis.wind_equations import *

def correct_and_save_light_curves(target_id_arr, observer_id_arr, data_file_arr, save_dir, name_arr, crop_range_arr=None):

    for i, data_file in enumerate(data_file_arr):
        time, raw, orbit_corrected, detrended = detrend_all(target_id_arr[i], observer_id_arr[i], data_file, 
                                                            crop_range=crop_range_arr[i] if crop_range_arr else None)
        np.savez(save_dir + f"{name_arr[i]}.npz", time=time, raw=raw, orbit_corrected=orbit_corrected, detrended=detrended)


def save_periodograms(sector_data_list, sector_data_strings, root, flux_type='detrended', 
                      minimum_frequency=1, maximum_frequency=3,
                      samples_per_peak=10):
    
    for i, sector_data in tqdm(enumerate(sector_data_list)):
        frequency, power, false_alarm_levels = make_periodogram(sector_data['time'], sector_data[flux_type],
                                                                minimum_frequency=minimum_frequency, 
                                                                maximum_frequency=maximum_frequency,
                                                                samples_per_peak=samples_per_peak)
        peaks, peak_pows = get_peak_frequencies(frequency, power, false_alarm_levels[0])
        # print(false_alarm_levels)
        np.savez(root + f'{sector_data_strings[i]}_periodogram.npz', 
                frequency=frequency, power=power, false_alarm_levels=false_alarm_levels,
                peaks=peaks, peak_pows=peak_pows)
        
        
def save_bootstrap(sector_data_list, sector_data_strings, root, flux_type='detrended', fap_level=0.1, min_period=0.1, max_period=1,
                   n_freqs=int(1e5), n_bootstraps=10000, plot=True):
    
    for i, sector_data in enumerate(sector_data_list):
        peak_periods = bootstrap_peak_periods(sector_data['time'], sector_data[flux_type], fap_level=fap_level, 
                                                min_period=min_period, max_period=max_period, n_freqs=n_freqs, 
                                                n_bootstraps=n_bootstraps, plot=plot)
        # labels, all_means, all_stds = cluster_peaks(peak_periods, eps=0.0001, plot=True, n_cols=3)
        np.savez(root + f'{sector_data_strings[i]}_bootstrap.npz', 
                 peak_periods=peak_periods)#, labels=labels, all_means=all_means, all_stds=all_stds)
        

def cluster_save(periodograms, peak_periods_list, sector_data_strings, save_dir, plot=False, eps=0.0001, tolerance= 0.005, ncols=3):
    for i, peak_periods in enumerate(peak_periods_list):
        _, all_means, all_stds = cluster_peaks(peak_periods, eps=eps, plot=plot, n_cols=ncols)
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


def mcmc_save(wind_eqns, wind_eqn_errs, cluster_arr, 
              Re, Rp, P, Re_err, Rp_err, P_err, 
              wind_eqn_strings, sector_data_string, root,
              min_freq_threshold=0.5):
    
    min_freq_arr = get_minimum_frequency_arr(wind_eqns, Re, Rp, P)
    model_eqn = RHS()

    phi_super_arr = []
    i = 0
    for wind_eqn, wind_eqn_err in zip(wind_eqns, wind_eqn_errs):
        freq_eqn = PHI(wind_eqn, Re, Rp, P) 
        net_sigma = sigma(wind_eqn, Re, Rp, P, wind_eqn_err, Re_err, Rp_err, P_err)

        if (min_freq_arr[i] > min_freq_threshold):
            min_freq = min_freq_arr[i] 
        else: 
            min_freq_arr[np.argmin(min_freq_arr[min_freq_arr > min_freq_threshold])]

        print(f"Using minimum frequency of {min_freq} for wind equation {wind_eqn_strings[i]}")
        period_limit = 1 / min_freq

        phi_arr = []

        means_filtered = 1 / cluster_arr['matched_means'][cluster_arr['matched_means'] < period_limit]
        stds_filtered_periods = cluster_arr['matched_stds'][cluster_arr['matched_means'] < period_limit]
        stds_filtered = stds_filtered_periods * (means_filtered**2)

        print("Processing wind equation:", wind_eqn_strings[i])
        for f_obs, f_err in zip(means_filtered, stds_filtered):
            print("Frequency, error:", f_obs, f_err)
            sampler = run_mcmc(f_obs, f_err, model_eqn, net_sigma, freq_eqn)

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