import numpy as np

from frequency_analysis.wind_equations import *
from frequency_analysis.mcmc import *
from frequency_analysis.save_data import *

root = '/home/ktp9/TESSNeptune24/tess-ice-giants/final_data/'

# load clustered peak data
u42_clustered = np.load(root + 'periodograms/u42_clustered_peaks.npz')
u43_clustered = np.load(root + 'periodograms/u43_clustered_peaks.npz')
u44_clustered = np.load(root + 'periodograms/u44_clustered_peaks.npz')
n42_clustered = np.load(root + 'periodograms/n42_clustered_peaks.npz')
n70_clustered = np.load(root + 'periodograms/n70_resampled_clustered_peaks.npz')

sector_data_strings = ['u42', 'u43', 'u44', 'n42', 'n70_resampled']   

R_nep = 24764 * 1000
P_nep = 15.9663
R_err_nep = 15000
P_err_nep = 0.0002

wind_eqns_nep = [sromovsky1993_four, sromovsky1993_six, tollefson2013_kp, tollefson2014_kp]
wind_eqn_errs_nep = [sromovsky1993_four_err, sromovsky1993_six_err, tollefson2013_kp_err, tollefson2014_kp_err]

R_ur = 25559 * 1000
P_ur = 17.247864
R_err_ur = 4000
P_err_ur = 0.00001

wind_eqn_ur = [sromovsky2012_odd_N, sromovsky2012_odd_S, sromovsky2015_N, sromovsky2015_S]
wind_eqn_errs_ur = [sigma_sromovsky2012_odd_N, sigma_sromovsky2012_odd_S, sigma_sromovsky2015_N, sigma_sromovsky2015_S]

for i, cluster in enumerate([n42_clustered, n70_clustered]): #u42_clustered, u43_clustered, u44_clustered, 
    i += 3  # TEMPORARY OVERRIDE FOR TESTING
    print("Processing sector data:", sector_data_strings[i])


    if i >= 3:
        mcmc_save(wind_eqns_nep, wind_eqn_errs_nep, cluster, 
                  R_nep, P_nep, R_err_nep, P_err_nep, 
                  ['sromovsky1993_four', 'sromovsky1993_six', 'tollefson2013_kp', 'tollefson2014_kp'],
                  f'{sector_data_strings[i]}', root + 'latitudes/')
    else:
        mcmc_save(wind_eqn_ur, wind_eqn_errs_ur, cluster, 
                  R_ur, P_ur, R_err_ur, P_err_ur, 
                  ['sromovsky2012_odd_N', 'sromovsky2012_odd_S', 'sromovsky2015_N', 'sromovsky2015_S'], 
                  f'{sector_data_strings[i]}', root + 'latitudes/')