from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import os
import random

#for colored graphs
from matplotlib.colors import Normalize
from matplotlib.pyplot import get_cmap

#for object lightcurve
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, aperture_photometry

from tqdm import tqdm

#variables for linelc
pltnum = 5 #number of plots
random_coordinates = [(114, 113), (51, 65), (79, 134), (89, 30)]#[(random.randint(0, 149), random.randint(0, 149)) for _ in range(pltnum -1)]

#diagnistic: takes random pixels and makes individual lightcurves for each
def linelc(arr, x, y, lcdir):
    
    flxxy = arr[x, y]

    # Generate the z indices
    z_indices = np.arange(flxxy.shape[0])

    #Plot the data
    plt.figure(figsize=(8, 2), dpi=300)
    plt.plot(z_indices, flxxy, linestyle='-', marker='')
    plt.xlabel('Cadence Number')
    plt.ylabel('Flux')
    #plt.ylim(-10, 10)
    #plt.xlim(2000, 4000)
    plt.title(f'Cadence vs. Malena Subtraction at ({x}, {y})')
    plt.grid(True)

    finnme = f"madiff_{x}_{y}.png"

    output = os.path.join(lcdir, finnme)
    plt.savefig(output)
    plt.close()
    #print(f"Plot saved as {output}")

#diagnistic: takes random pixels and charts lightcurves for each on the same plot
def scatterlc(arr, x_pixel_range, y_pixel_range, lcdir):
    
    pltnum = 200 #number of lightcurves on the graph
    random_coordinates = [(random.randint(x_pixel_range), random.randint(y_pixel_range)) for _ in range(pltnum)]

    # Generate the z indices
    z_indices = np.arange(arr.shape[2])

    #normalize coords to fit colormap
    norm = Normalize(vmin=0, vmax=149)
    cmap = plt.get_cmap('cool') #get creative!

    #Plot the data
    plt.figure(figsize=(7.5, 2.25), dpi = 300)
    for ii in tqdm(range(pltnum)):
        coord = random_coordinates[ii]
        color = cmap(norm(coord[1]))
        plt.plot(z_indices, arr[coord[0], coord[1], :], color=color, linestyle = '-', marker=' ')
    
    plt.xlabel('Cadence Number')
    plt.ylabel('Flux')
    #plt.ylim(-250, 10000)
    #plt.xlim(2000, 4000)
    plt.title(f'Cadence vs. Malena Subtraction over {pltnum} pixels')
    plt.grid(True)

    finnme = f"total_subtracted_lcs.png"

    output = os.path.join(lcdir, finnme)
    plt.savefig(output)
    plt.close() 

#pull out moving object position
def extract_positions(data_stack, times, position_last_x, position_last_y):
    
    positions_all = np.array([0,0])
    times_all = np.array([])
    #frames_mask = np.ones(len(data_stack[0,0]))
    nan_positions = np.array([np.nan, np.nan])

    for data_frame_num in tqdm(range(0, len(data_stack[0,0]))):
        data_frame = data_stack[:,:,data_frame_num]
        mean, median, std = sigma_clipped_stats(data_frame, sigma=3.0)
        threshold = np.percentile(data_frame, 99.99)
        #print(threshold)
        daofind = DAOStarFinder(fwhm=2.5, threshold=threshold)  

        sources = daofind(data_frame - median) 
        #print(f"Frame: {data_frame_num}, Sources found: {len(sources) if sources is not None else 0}")

        try:
            dist_last = np.sqrt(((sources['xcentroid'] - position_last_x)**2.) + (sources['ycentroid'] - position_last_y)**2.)
            #xdist_last = abs(sources['xcentroid'].value - position_last_x)
            
            where_source = np.argmin(dist_last)
            #print(f"Closest source distance: {dist_last[where_source]}")

            if dist_last[where_source] < 50:
   
                if abs(sources['xcentroid'].value - position_last_x) < 3:
                    positions = np.array([sources['xcentroid'][where_source], sources['ycentroid'][where_source]])
                    positions_all = np.vstack((positions_all, positions))
                    times_all = np.append(times_all, times[data_frame_num])

                    position_last_x, position_last_y = positions[0], positions[1]
                    #print(f"Appended position: {positions}, time: {times[data_frame_num]}")
       
                else:
                    #frames_mask[data_frame_num] = 0
                    positions_all = np.vstack((positions_all, nan_positions))
                    times_all = np.append(times_all, times[data_frame_num])
                    print(f"Index {data_frame_num} appended with NaNs. Source x-coordinate difference too large.")

            else:
                #frames_mask[data_frame_num] = 0
                positions_all = np.vstack((positions_all, nan_positions))
                times_all = np.append(times_all, times[data_frame_num])
                print(f"Index {data_frame_num} appended with NaNs. Source distance too large.")

        except:
            #frames_mask[data_frame_num] = 0
            positions_all = np.vstack((positions_all, nan_positions))
            times_all = np.append(times_all, times[data_frame_num])
            print(f"Index {data_frame_num} appended with NaNs. Exception occurred.")
    
    #write flags to fits file
    # flag_hdu = fits.ImageHDU(data=frames_mask, name='good_frame_flag')
    # hdu.append(flag_hdu)

    positions_all = positions_all[1:]
    
    return positions_all, times_all

# make lightcurve array
def get_summed_fluxes(data_stack, positions_all, r_aperture):

    summed_fluxes_all = np.array([])

    for frame_num in tqdm(range(0, len(data_stack[0,0]))):
        
        if np.any(np.isnan(positions_all[frame_num])):
            summed_fluxes_all = np.append(summed_fluxes_all, np.nan)

        else:
            data_frame = data_stack[:,:,frame_num]
    
            aperture = CircularAperture(positions_all[frame_num], r=r_aperture)#5)#1.5)
            phot_table = aperture_photometry(data_frame, aperture)
            summed_flux = phot_table['aperture_sum'].value[0]
            #print(positions_all[frame_num], summed_flux)
            summed_fluxes_all = np.append(summed_fluxes_all, summed_flux)
            
    return summed_fluxes_all

#UPDATE directories
lcdir = '/scratch11/ktp9/DIA/70/lcfiles/' #directory to save the lightcurve plots
fitdir = '/scratch11/ktp9/DIA/42/stacks/'

#example calls
#scatterlc(array, lcdir)
#for x, y in tqdm(random_coordinates):
    #linelc(array, x, y, lcdir)

# ---Make LC for planet---
#UPDATE filename
"""
data_fits = fitdir + 'sector42.fits'

with fits.open(data_fits) as hdu:
    raw_data = hdu[1].data
    data_stack = hdu[2].data
    times = hdu[3].data
"""
"""
startx = 138 #42: 127 #UPDATE
starty = 9 #42: 187 #UPDATE

data_stack = np.load('/scratch11/ktp9/DIA/70/sec70chunk30.npy')
times = np.load('/scratch11/ktp9/DIA/70/stacks/time.npy')

positions_all, times_all = extract_positions(data_stack, times, startx, starty)

positions_x_offset = positions_all.copy()
positions_x_offset = positions_x_offset[0] - 20

positions_y_offset = positions_all.copy()
positions_y_offset = positions_y_offset[1] - 20

print(f"Shape of positions: {positions_all.shape}")
print(f"Shape of times: {times_all.shape}")
#check for useable data count
mask = ~np.isnan(positions_all)
data_count = np.sum(mask, axis=0)
print(f"Data count: {data_count}")

summed_x_offset = get_summed_fluxes(data_stack, positions_x_offset)
summed_y_offset = (data_stack, positions_y_offset)

# for r in tqdm(range(4, 9)): #CONSIDER updating for your radius
#     summed_fluxes_all = get_summed_fluxes(data_stack, positions_all, r)
#     np.save(lcdir + "chunk30" + str(r) , summed_fluxes_all)
    
print("Here's your flux data! Toodle-loo, kagnaroo!")
"""