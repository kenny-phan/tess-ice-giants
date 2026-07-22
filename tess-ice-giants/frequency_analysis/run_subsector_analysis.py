import glob
import numpy as np

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

max_freq_arr = nyquist_from_cadence(sample_cadences)

total_segs = 10

times = [sector['time'] for sector in lc_list]
fluxes = [sector['orbit_corrected'] for sector in lc_list] 

fap_level=0.01
n_bootstraps=1000

# #this outputs orbit corrected time and flux; periodogram outputs after detrending
# subtimes, subfluxes, subfreqs, subpower, subfap, subpeaks = split_data(times, fluxes, 
#                                                                             max_freq_arr, freq_array_size=1000,
#                                                                             m=50, total_segs=total_segs, 
#                                                                             bootstrap=True,
#                                                                             verbose=True, fap_level=fap_level)

# np.savez(root + 'subsectors/subsectors.npz', subtimes=np.array(subtimes, dtype=object), subfluxes=np.array(subfluxes, dtype=object),
#          subfreqs=np.array(subfreqs, dtype=object), subpower=np.array(subpower, dtype=object), subfap=np.array(subfap, dtype=object),
#          subpeaks=np.array(subpeaks, dtype=object),
#          allow_pickle=True)

subsectors = np.load(root + 'subsectors/subsectors.npz', allow_pickle=True)
subtimes = list(subsectors['subtimes'])
subfluxes = list(subsectors['subfluxes'])
subfreqs = np.array(subsectors['subfreqs'], dtype=float)
subpower = list(subsectors['subpower'])
subfap = list(subsectors['subfap'])

# # bootstrap each periodogram
# max_freq_arr = nyquist_from_cadence(sample_cadences / 24)

# for sidx, sector in enumerate(planet_sectors):
#     print(f"bootstrapping {sector}")
#     boot_path = root + "subsectors/" + sector + "/bootstrap/"
#     Path(boot_path).mkdir(exist_ok=True)

#     subidx = 0
#     for time, flux in zip(subtimes[sidx], subfluxes[sidx]):
#         print(f"subsector {subidx}")
#         min_freq = 1/((time[-1] - time[0])/2)

#         peak_periods = bootstrap_peak_periods(time, flux, fap_level=fap_level, 
#                                                 min_period=1/max_freq_arr[sidx], 
#                                                 max_period=1/min_freq, 
#                                                 n_freqs=int(1e5), 
#                                                 n_bootstraps=n_bootstraps, plot=False)

#         np.savez(boot_path + f'{sector}_sub{subidx}_bootstrap.npz', 
#                  peak_periods=peak_periods)

#         subidx += 1

# cluster each bootstrap
eps=0.0001
tolerance= 0.005
ncols=3
plot=False

for sidx, sector in enumerate(planet_sectors):
    print(f"clustering {sector}")
    clust_path = root + "subsectors/" + sector + "/clusters/"
    Path(clust_path).mkdir(exist_ok=True)

    boot_path = root + "subsectors/" + sector + "/bootstrap/"

    subidx = 0

    # print(subfreqs.shape, len(subpower), len(subfap))
    for freq, pow, fap in zip(subfreqs[sidx], subpower[sidx], subfap[sidx]):
        print(f"subsector {subidx}")

        peak_periods = np.load(boot_path + f'{sector}_sub{subidx}_bootstrap.npz')["peak_periods"]
        _, all_means, all_stds = cluster_peaks(peak_periods, eps=eps, plot=plot, n_cols=ncols, n_bootstraps=n_bootstraps)
        
        peaks, _ = get_peak_frequencies(freq, pow, fap)

        peak_periodogram_periods = np.array(1/peaks)

        xmatch = np.abs(peak_periodogram_periods[:, np.newaxis] - np.array(all_means))
        potential_matches = np.abs(xmatch) < tolerance
        closest_matches, _ = np.unique(np.where(potential_matches)[1], return_counts=True)

        matched_means = np.array(all_means)[closest_matches]
        matched_stds = np.array(all_stds)[closest_matches]

        np.savez(clust_path + f'{sector}_sub{subidx}_clustered_peaks.npz', 
                 matched_means=matched_means, matched_stds=matched_stds)
        
        subidx+=1

# mcmc each cluster

# next lets import in some wind equations to interpret the frequencies we found
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

for sidx, sector in enumerate(planet_sectors):
    print(f"mcmc {sector}")
    clust_path = root + "subsectors/" + sector + "/clusters/"

    mcmc_path = root + "subsectors/" + sector + "/mcmc/"
    Path(mcmc_path).mkdir(exist_ok=True)

    cluster_list = glob.glob(clust_path + "*")

    for subidx, cluster in enumerate(cluster_list):

        if planet_sectors[sidx][0] == "u":
            save_mcmc(ur_wind_eqns, ur_wind_eqn_errs, cluster,
                    uRe, uRp, uP, uRe_err, uRp_err, uP_err, 
                    ur_wind_eqn_strings, f'{sector}_sub{subidx}', mcmc_path, reperrs=reperrs)
        elif planet_sectors[sidx][0] == "n":
            save_mcmc(nep_wind_eqns, nep_wind_eqn_errs, cluster, 
                    nRe, nRp, nP, nRe_err, nRp_err, nP_err, 
                    nep_wind_eqn_strings, f'{sector}_sub{subidx}', mcmc_path)

# save latitude solutions
# we now have posterior distributions for each solution, lets get one sigma interval ~10m
for sidx, sector in enumerate(planet_sectors):
    print(f"latsol {sector}")
    mcmc_path = root + "subsectors/" + sector + "/mcmc/"
    latsol_path = root + "subsectors/" + sector + "/latitudes/"

    Path(latsol_path).mkdir(exist_ok=True)

    mcmc_list = glob.glob(mcmc_path + "*")
    for i, phi_dist in enumerate(mcmc_list):
        print(f"Planet subsector: {i}")

        all_latitudes, all_standard_devs = fit_all_distributions(phi_dist["phi_distributions"], phi_dist["wind_eqn_strings"], plot=False)

        np.savez(latsol_path + f"{sector}_sub{subidx}_latitude_solutions.npz", 
                    lat=np.array(all_latitudes, dtype=object), std=np.array(all_standard_devs, dtype=object), allow_pickle=True)