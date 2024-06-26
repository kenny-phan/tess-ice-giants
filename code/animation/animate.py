from astropy.io import fits
import matplotlib.pyplot as plt
import os

import tqdm as tqdm

fitdir = '/'
pngdir = '/'
cdedir = '/'

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

    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()

    # iteratively adjust figure
    step = 10
    for n in range(0, len(data_stack[0,0])):

        if n%step == 0:
            # set panel 1 at each frame
            im.set_data(data_stack[:,:,n])
            #vmin = np.min(data_stack[:,:,n])
            #vmax = np.max(data_stack[:,:,n])
            vmin=0
            vmax=1000
            #im0.set_clim(vmin0, vmax0)
            im.set_clim(0, 1000)

            #cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Relative flux (e/s)')
            t = Time(times[n], format='jd').iso
            time_text.set_text(t)

            #frame_number.set_text('%i/%i Frames' %(n+1, len(data_stack[0,0])))
            plt.savefig(save_dir+'animation_frame%04d.png' %(n), dpi=800)


    filenames_ordered = sorted(glob.glob(save_dir+"*.png"), key=number)
    print(filenames_ordered)
    ims = [imageio.imread(f) for f in filenames_ordered] #glob.glob("*.png")]#sorted(glob.glob("*.png"), key=number)]
    imageio.mimwrite(save_dir+'animation.mp4', ims, fps=100)


fitdir = '/scratch11/ktp9/DIA/70/stacks/'
meddir = '/scratch11/ktp9/DIA/70/media/'

fits_filename = root_dir + 'stacks_Uranus_s42_cam4_ccd2_June2024.fits'
hdu = fits.open(fits_filename)
data_stack_bkgsub = hdu[2].data
times = hdu[3].data
sortind = np.argsort(times)
data_stack_bkgsub = data_stack_bkgsub[:,:,sortind]
times = times[sortind]

make_animation_1panel(data_stack_bkgsub, times, save_dir=root_dir+'Uranus_frames_s42_cam4_ccd2/')
