import numpy as np

from frequency_analysis.long_term_trends import bootstrap_peak_periods
from frequency_analysis.mcmc import *

root = '/home/ktp9/TESSNeptune24/tess-ice-giants/final_data/'
lc_dir = root + "light_curves/"
u42, u43, u44 = np.load(lc_dir+'u42_lcs.npz'), np.load(lc_dir+'u43_lcs.npz'), np.load(lc_dir+'u44_lcs.npz')
n42, n70 = np.load(lc_dir+'n42_lcs.npz'), np.load(lc_dir+'n70_lcs.npz')

#resample n70 to n42 time points
n70_times = group_and_average(n70['time'], n42['time'])
n70_corrections = group_and_average(n70['orbit_corrected'], n42['time'])
n70_detrended = group_and_average(n70['detrended'], n42['time'])

n70_resampled = {'time': n70_times, 'orbit_corrected': n70_corrections, 'detrended': n70_detrended}

sector_data_list = [u42, u43, u44, n42, n70_resampled]
sector_data_strings = ['u42', 'u43', 'u44', 'n42', 'n70_resampled']   

def save_bootstrap(sector_data_list, sector_data_strings, root):
    for i, sector_data in enumerate(sector_data_list):
        peak_periods = bootstrap_peak_periods(sector_data['time'], sector_data['detrended'], fap_level=0.1, 
                                                min_period=0.1, max_period=1, n_freqs=int(1e5), 
                                                n_bootstraps=10000, plot=True)
        # labels, all_means, all_stds = cluster_peaks(peak_periods, eps=0.0001, plot=True, n_cols=3)
        np.savez(root + f'periodograms/{sector_data_strings[i]}_bootstrap.npz', 
                 peak_periods=peak_periods)#, labels=labels, all_means=all_means, all_stds=all_stds)
        
save_bootstrap(sector_data_list, sector_data_strings, root)