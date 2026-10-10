import glob
import os
import re

import numpy as np
from tqdm import tqdm 

from figures import *
from subsectors import * 
from fullsector import nyquist_from_cadence
from bootstrap import bootstrap_peak_periods, cluster_peaks
from wind_equations import *
from mcmc import save_mcmc, fit_all_distributions

root = '/home/ktp9/TESSNeptune24/tess-ice-giants/'
data_dir = root + 'final_data/'

bps, pps = 50, 15
print(f"bps: {bps}, pps: {pps}")
bpps_dir = f'bps{bps}_pps{pps}/'

subdir = data_dir + f'subsectors/{bpps_dir}'

## intitialize the list of sectors 
planet_sectors = ["u42", "u43", "u44", "n42", "n70"]
sample_cadences = np.array([10/60, 10/60, 10/60, 10/60, (200/60)/60]) # in hours

supfigdir = root + "figures/supplementary/" + bpps_dir


# load in the light curves
lc_dir = data_dir + "light_curves/" + bpps_dir
lc_list = []
for sector in planet_sectors:
    lc_list.append(np.load(lc_dir + f'{sector}_lcs.npz'))

# ~10 min runtime, save subsectors

max_freq_arr = nyquist_from_cadence(sample_cadences / 24)

total_segs = 10

times = [sector['time'] for sector in lc_list]
fluxes = [sector['orbit_corrected'] for sector in lc_list]
result = split_data(times, fluxes, 
                    max_freq_arr, freq_array_size=10000,
                    m=50, total_segs=total_segs, 
                    bootstrap=True,
                    verbose=True)

(subtimes, subfluxes, subfreqs, subpower, subfap, subpeaks, subpeakfap) = result


os.makedirs(subdir, exist_ok=True)
print(f"saving to: {subdir}")
np.savez(subdir + 'subsectors.npz', 
         subtimes=np.array(subtimes, dtype=object), 
         subfluxes=np.array(subfluxes, dtype=object),
         subfreqs=np.array(subfreqs, dtype=object), 
         subpower=np.array(subpower, dtype=object), 
         subfap=np.array(subfap, dtype=object),
         subpeaks=np.array(subpeaks, dtype=object), 
         subpeakfap=np.array(subpeakfap, dtype=object),
         allow_pickle=True)

# load data back in 
subsectors = np.load(subdir + 'subsectors.npz', allow_pickle=True)
subtimes = subsectors['subtimes']
subfluxes = subsectors['subfluxes']
subfreqs = np.array(subsectors['subfreqs'], dtype=float)
subpower = subsectors['subpower']
subfap = subsectors['subfap']
subpeakfap = subsectors['subpeakfap']

# run bootstrap ~50 mins

nsec, nsub = subtimes.shape

bootstrap_results = np.empty((nsec, nsub), dtype=object)
for sec in range(nsec):
    print(f"Bootstrapping sector {sec+1}/{nsec}...")
    min_period=1/max_freq_arr[sec]

    for sub in range(nsub):
        print(f"Bootstrapping subsector {sub+1}/{nsub}...")
        max_period = (subtimes[sec, sub][-1] - subtimes[sec, sub][0])/2
        print(f"min, max period: {min_period, max_period}")

        peak_periods = bootstrap_peak_periods(subtimes[sec, sub], 
                                              subfluxes[sec, sub], 
                                              fap_level=0.01, 
                                              n_bootstraps=10000, 
                                              boot_percent=0.8, 
                                              min_period=min_period, 
                                              max_period=max_period, 
                                              n_freqs=1000, 
                                              figdir=supfigdir, 
                                              sds=planet_sectors[sec] + f"_ss{sub}",
                                              n_plot=100)
        bootstrap_results[sec, sub] = peak_periods

np.savez(subdir + 'bootstrap_results.npz', 
         bootstrap_results=bootstrap_results, 
         allow_pickle=True)

# # # run cluster ~10min
bootstrap_results = np.load(subdir + 'bootstrap_results.npz', 
                            allow_pickle=True)['bootstrap_results']

nsec, nsub = bootstrap_results.shape
cluster_results = np.empty((nsec, nsub), dtype=object)
for sec in range(nsec):
    print(f"Processing sector {sec}/{nsec}...")
    for sub in range(nsub):
        peaks = bootstrap_results[sec, sub]
        labels, all_means, all_stds = cluster_peaks(peaks, 
                                                    sds=planet_sectors[sec] + f"_ss{sub}",
                                                    eps=0.1, 
                                                    n_bootstraps=10000,
                                                    pass_frac=0.8, n_bins=50,
                                                    figdir=supfigdir + "clusters/")
        
        cluster_results[sec, sub] = (labels, all_means, all_stds)

np.savez(subdir + 'cluster_results.npz', 
         cluster_results=cluster_results, 
         allow_pickle=True)

# run mcmc
cluster_results = np.load(subdir + 'cluster_results.npz', allow_pickle=True)['cluster_results']
nsec, nsub = cluster_results.shape

ur_wind_eqns = [sromovsky2012_odd_N]#, sromovsky2012_odd_S, sromovsky2015_N, sromovsky2015_S]
# ur_wind_eqn_errs = [sigma_sromovsky2012_odd_N, sigma_sromovsky2012_odd_S, sigma_sromovsky2015_N, sigma_sromovsky2015_S]
ur_wind_eqn_errs = [sigma_uranus_model(wind_eqn) for wind_eqn in ur_wind_eqns]
ur_wind_eqn_strings = ["sromovsky2012_odd_N"]#, "sromovsky2012_odd_S", "sromovsky2015_N", "sromovsky2015_S"]

nep_wind_eqns = [sromovsky1993_four]#, sromovsky1993_six, tollefson2013_kp, tollefson2014_kp]
nep_wind_eqn_errs = [sromovsky1993_four_err]#, sromovsky1993_six_err, tollefson2013_kp_err, tollefson2014_kp_err]
nep_wind_eqn_strings = ["sromovsky1993_four"]#, "sromovsky1993_six", "tollefson2013_kp", "tollefson2014_kp"]

# planet data
uRe = 25559 * 1000
uRp = 24973 * 1000
uP = 17.247864
uRe_err = 4000
uRp_err = 20000
uP_err = 0.00001

nRe = 24764 * 1000
nRp = 24341 * 1000
nP = 15.9663
nRe_err = 15000
nRp_err = 30000
nP_err = 0.0002

reperr12 = 0.088 # degrees/h, pg. 11 of Sromovsky+ 2012c
reperr15 = 0.147 / 24 # 0.147 degrees/day, pg. 11 of Sromovsky+ 2015
reperrs = [reperr12, reperr12, reperr15, reperr15]

sub_root = root + "subsectors/"

nsec, nsub = cluster_results.shape

mcmc_results = np.empty((nsec, nsub))
for i, sec in enumerate(range(nsec)):
    mcmc_root = subdir + f"{planet_sectors[i]}/"

    for sub in range(nsub):
        labels, all_means, all_stds = cluster_results[sec, sub]
        sector_data = {}
        sector_data['matched_means'] = all_means
        sector_data['matched_stds'] = all_stds
        
        if planet_sectors[i][0] == "u":
            print("Uranus sector: ", planet_sectors[i], "sub", sub)
            save_mcmc(ur_wind_eqns, ur_wind_eqn_errs, sector_data,
                        uRe, uRp, uP, uRe_err, uRp_err, uP_err, 
                        ur_wind_eqn_strings, f"subsector{sub}", mcmc_root + "mcmc/", reperrs=reperrs)
        elif planet_sectors[i][0] == "n":
            print("Neptune sector: ", planet_sectors[i], "sub", sub)
            save_mcmc(nep_wind_eqns, nep_wind_eqn_errs, sector_data, 
                        nRe, nRp, nP, nRe_err, nRp_err, nP_err, 
                        nep_wind_eqn_strings, f"subsector{sub}", mcmc_root + "mcmc/")

# # we now have posterior distributions for each solution, lets get one sigma interval ~20m

subroot = root + "subsectors"

for sector in planet_sectors:
    secsubroot = subdir + "/" + sector
    mcmc_list = glob.glob(secsubroot + "/mcmc/*")
    mcmc_list.sort(key=lambda x: int(re.search(r'subsector(\d+)', x).group(1)))

    for i, phi_file in enumerate(mcmc_list):

        phi_dist = np.load(phi_file, allow_pickle=True)
        phi_data = phi_dist['phi_distributions']

        all_latitudes, all_standard_devs = fit_all_distributions(phi_dist["phi_distributions"], 
                                                                        phi_dist["wind_eqn_strings"], 
                                                                        verbose=False, 
                                                                        n_bins=50, 
                                                                        floor_frac=0.01,
                                                                        truncbound=None, 
                                                                        sds=sector, 
                                                                        figdir=supfigdir + "subposteriors/")

        print(all_latitudes, all_standard_devs)

        if os.path.exists(secsubroot + "/latitudes/") == False:
            os.makedirs(secsubroot + "/latitudes/")
        np.savez(secsubroot + "/latitudes/" + f"{sector}_sub{i}_latitude_solutions.npz", 
                    lat=np.array(all_latitudes, dtype=object), 
                    std=np.array(all_standard_devs, dtype=object), 
                    allow_pickle=True) 
