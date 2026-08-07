import glob
import os

import numpy as np
from tqdm import tqdm 

from pathlib import Path

from figures import *
from subsectors import * 
from fullsector import nyquist_from_cadence
from bootstrap import bootstrap_peak_periods, cluster_peaks
from wind_equations import *
from mcmc import save_mcmc, fit_all_distributions

root = '/home/ktp9/TESSNeptune24/tess-ice-giants/final_data/'

## intitialize the list of sectors 
planet_sectors = ["u42", "u43", "u44", "n42", "n70"]
sample_cadences = np.array([10/60, 10/60, 10/60, 10/60, (200/60)/60]) # in hours

# load in the light curves
lc_dir = root + 'light_curves/'
lc_list = []
for sector in planet_sectors:
    lc_list.append(np.load(lc_dir + f'{sector}_lcs.npz', allow_pickle=True))

# ~10 min runtime, save subsectors
# max_freq_arr = nyquist_from_cadence(sample_cadences)

# total_segs = 10

# times = [sector['time'] for sector in lc_list]
# fluxes = [sector['orbit_corrected'] for sector in lc_list]
# subtimes, subfluxes, subfreqs, subpower, subfap, subpeaks = split_data(times, fluxes, 
#                                                                             max_freq_arr, freq_array_size=1000,
#                                                                             m=50, total_segs=total_segs, 
#                                                                             bootstrap=True,
#                                                                             verbose=True)

# np.savez(root + 'subsectors/subsectors.npz', subtimes=np.array(subtimes, dtype=object), subfluxes=np.array(subfluxes, dtype=object),
#          subfreqs=np.array(subfreqs, dtype=object), subpower=np.array(subpower, dtype=object), subfap=np.array(subfap, dtype=object),
#          subpeaks=np.array(subpeaks, dtype=object),
#          allow_pickle=True)

# run bootstrap ~50 mins
subsectors = np.load(root + 'subsectors/subsectors.npz', allow_pickle=True)
subtimes = subsectors['subtimes']
subfluxes = subsectors['subfluxes']
subfreqs = np.array(subsectors['subfreqs'], dtype=float)
subpower = subsectors['subpower']
subfap = subsectors['subfap']

nsec, nsub = subtimes.shape

bootstrap_results = np.empty((nsec, nsub), dtype=object)
for sec in range(nsec):
    print(f"Bootstrapping sector {sec+1}/{nsec}...")
    for sub in range(nsub):
        print(f"Bootstrapping subsector {sub+1}/{nsub}...")
        peak_periods = bootstrap_peak_periods(subtimes[sec, sub], 
                                              subfluxes[sec, sub], 
                                              fap_level=0.01, 
                                              n_bootstraps=10000, boot_percent=0.8, 
                                              min_period=5/24, max_period=25/24, 
                                              n_freqs=1000, plot=True, n_plot=100)
        bootstrap_results[sec, sub] = peak_periods

np.savez(root + 'subsectors/bootstrap_results.npz', bootstrap_results=bootstrap_results, allow_pickle=True)

# # run cluster ~10min
nsec, nsub = bootstrap_results.shape
cluster_results = np.empty((nsec, nsub), dtype=object)
for sec in range(nsec):
    print(f"Processing sector {sec}/{nsec}...")
    for sub in range(nsub):
        peaks = bootstrap_results[sec, sub]
        labels, all_means, all_stds = cluster_peaks(peaks, 
                                                    eps=0.005, 
                                                    plot=True,
                                                    allow_skew_truc=True,
                                                    skew_threshold=0.9,
                                                    min_prominence=0.8, 
                                                    n_bootstraps=10000,
                                                    pass_frac=0.8)
        cluster_results[sec, sub] = (labels, all_means, all_stds)

np.savez(root + 'subsectors/cluster_results.npz', cluster_results=cluster_results, allow_pickle=True)

# run mcmc
cluster_results = np.load(root + 'subsectors/cluster_results.npz', allow_pickle=True)['cluster_results']

ur_wind_eqns = [sromovsky2012_odd_N, sromovsky2012_odd_S, sromovsky2015_N, sromovsky2015_S]
# ur_wind_eqn_errs = [sigma_sromovsky2012_odd_N, sigma_sromovsky2012_odd_S, sigma_sromovsky2015_N, sigma_sromovsky2015_S]
ur_wind_eqn_errs = [sigma_uranus_model(wind_eqn) for wind_eqn in ur_wind_eqns]
ur_wind_eqn_strings = ["sromovsky2012_odd_N", "sromovsky2012_odd_S", "sromovsky2015_N", "sromovsky2015_S"]

nep_wind_eqns = [sromovsky1993_four, sromovsky1993_six, tollefson2013_kp, tollefson2014_kp]
nep_wind_eqn_errs = [sromovsky1993_four_err, sromovsky1993_six_err, tollefson2013_kp_err, tollefson2014_kp_err]
nep_wind_eqn_strings = ["sromovsky1993_four", "sromovsky1993_six", "tollefson2013_kp", "tollefson2014_kp"]

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

sub_root = root + "/subsectors/"

nsec, nsub = cluster_results.shape

mcmc_results = np.empty((nsec, nsub))
for i, sec in enumerate(range(nsec)):
    mcmc_root = sub_root + f"{planet_sectors[i]}/"

    for sub in range(nsub):
        labels, all_means, all_stds = cluster_results[sec, sub]
        for i, std in enumerate(all_stds):
            try:
                if len(std) > 1:
                    all_stds[i] = np.mean(std)
            except TypeError:
                # std is a scalar, leave it alone
                pass

        sector_data = {}
        sector_data['matched_means'] = []
        sector_data['matched_stds'] = []
        
        if len(all_means) > 0:
            sector_data['matched_means'].extend(all_means)
            sector_data['matched_stds'].extend(all_stds)
        
        # for key in sector_data:
        #     sector_data[key] = np.array(sector_data[key])

        print(f"Type of sector_data['matched_means']: {type(sector_data['matched_means'])}")
        print(f"Dtype: {sector_data['matched_means'].dtype if hasattr(sector_data['matched_means'], 'dtype') else 'No dtype'}")
        if planet_sectors[i][0] == "u":
            print("Uranus sector: ", planet_sectors[i])
            save_mcmc(ur_wind_eqns, ur_wind_eqn_errs, sector_data,
                        uRe, uRp, uP, uRe_err, uRp_err, uP_err, 
                        ur_wind_eqn_strings, f"subsector{sub}", mcmc_root + "mcmc/", reperrs=reperrs)
        elif planet_sectors[i][0] == "n":
            print("Neptune sector: ", planet_sectors[i])
            save_mcmc(nep_wind_eqns, nep_wind_eqn_errs, sector_data, 
                        nRe, nRp, nP, nRe_err, nRp_err, nP_err, 
                        nep_wind_eqn_strings, f"subsector{sub}", mcmc_root + "mcmc/")

# # we now have posterior distributions for each solution, lets get one sigma interval ~20m

subroot = root + "subsectors"

print(root)
for sector in planet_sectors:
    secsubroot = subroot + "/" + sector
    mcmc_list = glob.glob(secsubroot + "/mcmc/*")

    for i, phi_file in tqdm(enumerate(mcmc_list)):
        phi_dist = np.load(phi_file, allow_pickle=True)

        # print(len(phi_dist["phi_distributions"][0]))
        # print(f"Planet sector: {planet_sectors[i]}")

        all_latitudes, all_standard_devs = fit_all_distributions(phi_dist["phi_distributions"], 
                                                                phi_dist["wind_eqn_strings"], 
                                                                plot=False)

        if os.path.exists(secsubroot + "/latitudes/") == False:
            os.makedirs(secsubroot + "/latitudes/")

        np.savez(secsubroot + "/latitudes/" + f"{sector}_sub{i}_latitude_solutions.npz", 
                    lat=np.array(all_latitudes, dtype=object), 
                    std=np.array(all_standard_devs, dtype=object), 
                    allow_pickle=True) #   mcmc_list.append(np.load(mcmc_dir + f"{sector}_phi_distributions.npz", allow_pickle=True))
