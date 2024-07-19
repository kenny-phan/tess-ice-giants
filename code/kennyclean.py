# based on Rice 2024
import numpy as np

def subtract_background(data_stack, size_cutout_side, percent_pix_scatteredlight=5):

    # get a scattered light profile to subtract from all pixels

    # find median flux of each pixel
    median_flux_time = np.median(data_stack, axis=2) # (N_pix x N_pix grid of median fluxes)
    print(f"median_flux_time shape = {median_flux_time.shape}")

    # take just the pixels with medians in the the lowest 5th percentile
    flux_max = np.percentile(median_flux_time, percent_pix_scatteredlight)
    ind1_lowmedianflux, ind2_lowmedianflux = np.where(median_flux_time < flux_max)
    pixel_profiles_Nth_percentile = data_stack[ind1_lowmedianflux, ind2_lowmedianflux, :]
    print(f"pixel_profiles_Nth shape = {pixel_profiles_Nth_percentile.shape}")

    # take the median profile of the pixels in the lowest 5th percentile
    scattered_light_profile = np.median(pixel_profiles_Nth_percentile, axis=0)
    print(f"scattered_light_profile shape = {scattered_light_profile.shape}")

    # tile this and median profile
    tiled_scattered_light = np.tile(scattered_light_profile, (size_cutout_side, size_cutout_side, 1))
    print(f"tiled_scattered_light shape = {tiled_scattered_light.shape}")

    tiled_median_flux = np.repeat(median_flux_time[:, :, np.newaxis], len(data_stack[0,0]), axis=2)
    median_scattered_light_profile = tiled_scattered_light + (tiled_median_flux - np.median(tiled_scattered_light))
    print(f"tiled_median_flux shape = {tiled_median_flux.shape}")
    print(f"median_scattered_light shape = {median_scattered_light_profile.shape}")
    
    # subtract off this background profile from the data stack
    data_stack_tiled_sub = data_stack - median_scattered_light_profile
    print(f"data_stack_tiled_sub shape = {data_stack_tiled_sub.shape}")

    return data_stack_tiled_sub

# Function to process the array in chunks
def chunk(arr, chunk_size, percent_pix_scatteredlight=5):
    D0, D1, D2 = arr.shape
    output_array = np.zeros_like(arr)
    
    for i in range(0, D0, chunk_size):
        for j in range(0, D1, chunk_size):
            # Define the chunk boundaries
            chunk_d0_end = min(i + chunk_size, D0)
            chunk_d1_end = min(j + chunk_size, D1)
            
            # Extract the chunk
            chunk = arr[i:chunk_d0_end, j:chunk_d1_end, :]
            
            # Process the chunk
            subtracted_chunk = subtract_background(chunk, chunk_size)
            
            # Put the processed chunk back into the output array
            output_array[i:chunk_d0_end, j:chunk_d1_end, :] = subtracted_chunk
    
    return output_array

#UPDATE directories
stkfil = '/scratch11/ktp9/DIA/70/stacks/raw.npy'
optfil = '/scratch11/ktp9/DIA/70/sec70chunk30.npy'
raw = np.load(stkfil)
#axs = 256 #UPDATE with axis size in pixles
chunk_size = 30 #how big do you want the sample boxes?
array = chunk(raw, chunk_size)
np.save(optfil, array)
print("All done! Hit the road, happy taod!")
