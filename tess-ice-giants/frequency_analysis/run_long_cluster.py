import numpy as np
from bootstrap import save_cluster

# load in data
root = "/home/ktp9/TESSNeptune24/tess-ice-giants/final_data/"

## intitialize the list of sectors 
planet_sectors = ["u42", "u43", "u44", "n42", "n70"]

periodogram_list = []
bootstrap_list = []
for sector in planet_sectors:
    periodogram_list.append(np.load(root + "periodograms/" + f"{sector}_periodogram.npz"))
    bootstrap_list.append(np.load(root + "bootstrap/" + f"{sector}_bootstrap.npz")["peak_periods"])

save_cluster(periodogram_list, 
             bootstrap_list, 
             [f"{sector}" for sector in planet_sectors], 
             root + "long_term_clusters/",
             eps_arr=[0.1, 0.1, 0.1, 0.1, 0.1], tolerance= 0.5,
             n_cols=3, min_prominence_arr=[0.5, 0.5, 0.5, 0.4, 5],
             plot=True, verbose=True)

cluster_list = []
for sector in planet_sectors:
    cluster_list.append(np.load(root + "long_term_clusters/" + f"{sector}_clustered_peaks.npz", allow_pickle=True))
for i, cluster in enumerate(cluster_list):
    print(f"Sector {planet_sectors[i]}: {len(cluster['matched_means'])} clusters found.")