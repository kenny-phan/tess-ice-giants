import numpy as np

def subtract_background(data_stack, size_cutout_side, percent_pix_scatteredlight=5):

    # get a scattered light profile to subtract from all pixels

    # find median flux of each pixel
    median_flux_time = np.median(data_stack, axis=2) # (N_pix x N_pix grid of median fluxes)

    # take just the pixels with medians in the the lowest 5th percentile
    flux_max = np.percentile(median_flux_time, percent_pix_scatteredlight)
    ind1_lowmedianflux, ind2_lowmedianflux = np.where(median_flux_time < flux_max)
    pixel_profiles_Nth_percentile = data_stack[ind1_lowmedianflux, ind2_lowmedianflux, :]

    # take the median profile of the pixels in the lowest 5th percentile
    scattered_light_profile = np.median(pixel_profiles_Nth_percentile, axis=0)

    # tile this and median profile
    tiled_scattered_light = np.tile(scattered_light_profile, (size_cutout_side, size_cutout_side, 1))
    tiled_median_flux = np.repeat(median_flux_time[:, :, np.newaxis], len(data_stack[0,0]), axis=2)
    median_scattered_light_profile = tiled_scattered_light + (tiled_median_flux - np.median(tiled_scattered_light))
    
    # subtract off this background profile from the data stack
    data_stack_tiled_sub = data_stack - median_scattered_light_profile

    return data_stack_tiled_sub

stkfil = '/scratch11/ktp9/DIA/70/stacks/raw.npy'
optfil = '/scratch11/ktp9/DIA/70/stacks/mcleaned.npy'
raw = np.load(stkfil)
axs = 150
array = subtract_background(raw, axs)
np.save(optfil, array)
print("All done! In a while, crocodeil!")
