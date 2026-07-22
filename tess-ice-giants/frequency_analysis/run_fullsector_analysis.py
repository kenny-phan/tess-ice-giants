# base imports
import numpy as np
import glob 

from orbit_correction import correct_and_save_light_curves
from fullsector import save_periodograms, nyquist_from_cadence
from bootstrap import save_bootstrap, save_cluster
from wind_equations import *
from mcmc import save_mcmc, fit_all_distributions

root = '/home/ktp9/TESSNeptune24/tess-ice-giants/final_data/'

## intitialize the list of sectors 
planet_sectors = ["u42", "u43", "u44", "n42", "n70"]
sample_cadences = np.array([10/60, 10/60, 10/60, 10/60, (200/60)/60]) # in hours
crop_range_arr = [[1400, 1600], [], [], [975, 1040], []] # in pixels

# orbit and detrend processing (~2mins) 
# !! actually, orbit correction is already applied in the 
# !! tess_solarsystem pipeline. this simply detrends the data 

# ## here we pull light curves from the tess_solarsystem_planets pipeline
# raw_light_curves = f"{root}raw_light_curves/"

# uranus_id = '799'  
# neptune_id = '899'         
# observer_id = '@tess'  

# target_id_arr = [uranus_id, uranus_id, uranus_id, neptune_id, neptune_id]
# observer_id_arr = [observer_id]*5

# # uranus first, then neptune
# ur_raw_lcs = sorted(glob.glob(raw_light_curves + 'Uranus*/lc_Uranus*.txt'))
# nep_raw_lcs = sorted(glob.glob(raw_light_curves + 'Neptune*/lc_Neptune*.txt'))
# data_file_arr = ur_raw_lcs + nep_raw_lcs

# correct_and_save_light_curves(
#     target_id_arr, 
#     observer_id_arr, 
#     data_file_arr, 
#     root + 'light_curves/', 
#     name_arr = [f"{sector}_lcs" for sector in planet_sectors],
#     crop_range_arr = crop_range_arr
# )   

# now, lets find the peak frequencies (~30m)

lc_dir = root + "light_curves/"
lc_list = []
for sector in planet_sectors:
    lc_list.append(np.load(lc_dir + f"{sector}_lcs.npz"))

sector_baselines = [(lc['time'][-1] - lc['time'][0]) for lc in lc_list]

min_freq_arr = 1/(np.array(sector_baselines)/2)
max_freq_arr = nyquist_from_cadence(sample_cadences / 24)

# # make and save the periodograms
# periodogram_dir = root + "periodograms/"
# save_periodograms(lc_list, 
#                   [f"{sector}" for sector in planet_sectors], 
#                   periodogram_dir, fap_idx=1, 
#                   min_freq_arr=min_freq_arr, 
#                   max_freq_arr=max_freq_arr)

# ## now we bootstrap to get uncertainties on the peak frequencies (~100m)
    
# save_bootstrap(lc_list, 
#                [f"{sector}" for sector in planet_sectors], 
#                root + "bootstrap/", fap_level=1/100,
#                min_period_arr=1/max_freq_arr,
#                max_period_arr=1/min_freq_arr)

# # now, cluster the bootstrapped periodograms with dbscan (30s)
# from bootstrap import save_cluster

periodogram_list = []
bootstrap_list = []
for sector in planet_sectors:
    periodogram_list.append(np.load(root + "periodograms/" + f"{sector}_periodogram.npz"))
    bootstrap_list.append(np.load(root + "bootstrap/" + f"{sector}_bootstrap.npz")["peak_periods"])

# save_cluster(periodogram_list, 
#              bootstrap_list, 
#              [f"{sector}" for sector in planet_sectors], 
#              root + "clusters/",
#              eps_arr=[0.001, 0.001, 0.001, 0.001, 0.005], tolerance= 0.005,
#              n_cols=3, min_prominence_arr=[0.5, 0.5, 0.5, 0.4, 5],
#              plot=False)

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

# run mcmc fits for each cluster and save the results (~115m)
# load in the clusters

cluster_list = []
for sector in planet_sectors:
    cluster_list.append(np.load(root + "clusters/" + f"{sector}_clustered_peaks.npz", allow_pickle=True))

for i, cluster in enumerate(cluster_list):
    # frequencies = periodogram['peaks']
    # frequency_errs = periodogram['peak_std']

    if planet_sectors[i][0] == "u":
        print("Uranus sector: ", planet_sectors[i])
        # save_mcmc(ur_wind_eqns, ur_wind_eqn_errs, cluster,
        #           uRe, uRp, uP, uRe_err, uRp_err, uP_err, 
        #           ur_wind_eqn_strings, f"{planet_sectors[i]}", root + "mcmc/", reperrs=reperrs)
    elif planet_sectors[i][0] == "n":
        print("Neptune sector: ", planet_sectors[i])
        save_mcmc(nep_wind_eqns, nep_wind_eqn_errs, cluster, 
                  nRe, nRp, nP, nRe_err, nRp_err, nP_err, 
                  nep_wind_eqn_strings, f"{planet_sectors[i]}", root + "mcmc/")

# load the mcmc posteriors back in
mcmc_dir = root + "mcmc/"

mcmc_list = []
for sector in planet_sectors:
    mcmc_list.append(np.load(mcmc_dir + f"{sector}_phi_distributions.npz", allow_pickle=True))

# we now have posterior distributions for each solution, lets get one sigma interval ~10m

for i, phi_dist in enumerate(mcmc_list):
    print(len(phi_dist["phi_distributions"][0]))
    print(f"Planet sector: {planet_sectors[i]}")

    all_latitudes, all_standard_devs = fit_all_distributions(phi_dist["phi_distributions"], 
                                                             phi_dist["wind_eqn_strings"], 
                                                             plot=False)

    np.savez(root + "latitudes/" + f"{planet_sectors[i]}_latitude_solutions.npz", 
                lat=np.array(all_latitudes, dtype=object), 
                std=np.array(all_standard_devs, dtype=object), 
                allow_pickle=True)

