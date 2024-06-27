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
def scatterlc(arr, lcdir):
    
    pltnum = 200 #number of lightcurves on the graph
    random_coordinates = [(random.randint(125, 149), random.randint(0, 149)) for _ in range(pltnum)]

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
    plt.ylim(-250, 10000)
    #plt.xlim(2000, 4000)
    plt.title(f'Cadence vs. Malena Subtraction over {pltnum} pixels')
    plt.grid(True)

    finnme = f"total_subtracted_lcs.png"

    output = os.path.join(lcdir, finnme)
    plt.savefig(output)
    plt.close() 

#pull out moving object position
def extract_positions(data_fits, position_last_x, position_last_y):

    with fits.open(data_fits) as hdu:
    data_stack = hdu[2].data
    times = hdu[3].data
    
    positions_all = np.array([0,0])
    times_all = np.array([])
    frames_mask = np.ones(len(data_stack[0,0]))

    for data_frame_num in range(0, len(data_stack[0,0])):
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
            print(f"Closest source distance: {dist_last[where_source]}")

            if dist_last[where_source] < 50:
   
                if abs(sources['xcentroid'].value - position_last_x) < 3:
                    positions = np.array([sources['xcentroid'][where_source], sources['ycentroid'][where_source]])
                    positions_all = np.vstack((positions_all, positions))
                    times_all = np.append(times_all, times[data_frame_num])

                    position_last_x, position_last_y = positions[0], positions[1]
                    #print(f"Appended position: {positions}, time: {times[data_frame_num]}")
       
                else:
                    frames_mask[data_frame_num] = 0
                    print("Position appended with NaNs. Source x-coordinate difference too large.")

            else:
                frames_mask[data_frame_num] = 0
                print("Position appended with NaNs. Source distance too large.")

        except:
            frames_mask[data_frame_num] = 0
            print("Position appended with NaNs. Exception occurred.")
    
    #write flags to fits file
    # flag_hdu = fits.ImageHDU(data=frames_mask, name='good_frame_flag')
    # hdu.append(flag_hdu)

    positions_all = positions_all[1:]
    
    return positions_all, times_all

# make lightcurve array
def get_summed_fluxes(data_fits, positions_all, r_aperture):

    with fits.open(data_fits) as hdu:
        data_stack = hdu[2].data
        times = hdu[3].data
        #good_frame_flag = hdu[-1].data
        
    reduced_image_stack_pos = 
    
    summed_fluxes_all = np.array([])

    for frame_num in range(0, len(reduced_image_stack_pos[0,0])):

        data_frame = reduced_image_stack_pos[:,:,frame_num]

        aperture = CircularAperture(positions_all[frame_num], r=r_aperture)#5)#1.5)
        phot_table = aperture_photometry(data_frame, aperture)
        summed_flux = phot_table['aperture_sum'].value[0]
        #print(positions_all[frame_num], summed_flux)
        summed_fluxes_all = np.append(summed_fluxes_all, summed_flux)
        
    return summed_fluxes_all

# #plot to png
# def makelc(data, aperture_range):
    
#     for i in aperture_range:
#         get_summed_fluxes(data, )

lcdir = '/scratch11/ktp9/DIA/70/lcfiles/' #directory to save the lightcurve plots
fitdir = '/scratch11/ktp9/DIA/70/stacks/'

#example calls
#scatterlc(array, lcdir)
#for x, y in tqdm(random_coordinates):
    #linelc(array, x, y, lcdir)

data_fits = fitdir + 'sector70.fits'

startx = 135
starty = 15
positions_all, times_all = extract_positions(data_fits, startx, starty)
print(f"Shape of positions: {positions_all.shape}")
print(f"Shape of times: {times_all.shape}")

# r_aperture = 6
# summed_fluxes_all = get_summed_fluxes(reduced_image_stack-pos, positions_all, )

print("Here's your plot! Toodle-loo, kagnaroo!")
