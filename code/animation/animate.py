from astropy.io import fits
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy.time import Time
from mpl_toolkits.axes_grid1 import make_axes_locatable
import imageio.v2 as imageio
import os, glob, natsort
from os import listdir

from tqdm import tqdm

def make_animation_1panel(data_stack, times, save_dir='./'):

    if os.path.isdir(save_dir) == False:
        os.mkdir(save_dir)

    fig, ax = plt.subplots(1, 1, sharey=True, figsize=(4, 4))

    # panel 1
    im = ax.imshow(data_stack[:,:,0], origin='lower', vmin=0, vmax=1000)
    time_text = ax.text(1.0, 1.0, '', size=10, color='white')
    #frame_number0 = ax0.text(92.0, 120.0, '', size=10, color='white')
    #frame_number = ax.text(1.0, 245.0, '', size=10, color='white')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.1)
    cbar = fig.colorbar(im, ax=ax, cax=cax, label='Relative flux (e/s)')

    # set plot features
    ax.set_ylim(0, len(data_stack))

    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()

    # iteratively adjust figure
    step = 10
    for n in tqdm(range(0, len(data_stack[0,0]))):

        if n%step == 0:
            # set panel 1 at each frame
            im.set_data(data_stack[:,:,n])
            vmin=0
            vmax=1000
            im.set_clim(0, 1000)

            #cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Relative flux (e/s)')
            t = Time(times[n]+2457000, format='jd').iso
            time_text.set_text(t)

            #frame_number.set_text('%i/%i Frames' %(n+1, len(data_stack[0,0])))
            plt.savefig(save_dir+'animation_frame%05d.png' %(n), dpi=800)
    
    print("Writing images to video...")
    filenames_ordered = natsort.natsorted(glob.glob(save_dir+"*.png"))
    ims = [imageio.imread(f) for f in filenames_ordered]
    imageio.mimwrite(os.path.join(save_dir, 'dirty.mp4'), ims, fps=100)

    #remove leftover pngs
    delete = [f for f in glob.glob('*.png') if isfile(join(save_dir, f))]
    for file in delete:
        os.remove(file)

fitdir = '/scratch11/ktp9/DIA/70/stacks/'
meddir = '/scratch11/ktp9/DIA/70/media/'

fits_filename = fitdir + 'sector70.fits'
with fits.open(fits_filename) as hdu:
    data_stack_bkgsub = hdu[1].data
    times = hdu[3].data
    sortind = np.argsort(times)
    data_stack_bkgsub = data_stack_bkgsub[:,:,sortind]
    times = times[sortind]

make_animation_1panel(data_stack_bkgsub, times, save_dir=meddir)
print("Animation made! Bye-bye buttrefly!")
