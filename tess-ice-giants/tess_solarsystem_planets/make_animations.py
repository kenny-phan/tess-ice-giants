from astropy.io import fits
import numpy as np
from pylab import *
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy.time import Time
from mpl_toolkits.axes_grid1 import make_axes_locatable
import imageio
import os
from astropy.timeseries import LombScargle


dpi = 400  # 800


def number(filename):
    return int(filename[-8:-4])


def make_animation_1panel(data_stack, times, save_dir="./"):

    vmin_set = 0
    vmax_set = 200

    if os.path.isdir(save_dir) == False:
        os.mkdir(save_dir)

    fig, ax = plt.subplots(1, 1, sharey=True, figsize=(4, 4))

    # panel 1
    im = ax.imshow(data_stack[:, :, 0], origin="lower", vmin=vmin_set, vmax=vmax_set)
    time_text = ax.text(1.0, 1.0, "", size=10, color="white")
    # frame_number0 = ax0.text(92.0, 120.0, '', size=10, color='white')
    # frame_number = ax.text(1.0, 245.0, '', size=10, color='white')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.1)
    cbar = fig.colorbar(im, ax=ax, cax=cax, label="Relative flux (e/s)")

    # set plot features
    ax.set_ylim(0, len(data_stack))

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()

    # iteratively adjust figure

    for n in range(0, len(data_stack[0, 0])):

        if n % 5 == 0:
            # set panel 1 at each frame
            im.set_data(data_stack[:, :, n])

            im.set_clim(vmin_set, vmax_set)

            t = Time(times[n], format="jd").iso
            time_text.set_text(t)

            # frame_number.set_text('%i/%i Frames' %(n+1, len(data_stack[0,0])))
            plt.savefig(save_dir + "animation_frame%04d.png" % (n), dpi=800)

    filenames_ordered = sorted(glob.glob(save_dir + "*.png"), key=number)
    # print(filenames_ordered)
    # ims = [imageio.imread(f) for f in filenames_ordered]
    # imageio.mimwrite(save_dir+'animation.mp4', ims, fps=100)

    with imageio.get_writer("animation.mp4", mode="I") as writer:
        for filename in filenames_ordered:
            image = imageio.imread(filename)
            writer.append_data(image)

    plt.clf()
    plt.close()


def make_animation_2panel(
    data_stack_1,
    data_stack_2,
    times,
    root_dir,
    vmin1,
    vmax1,
    vmin2,
    vmax2,
    loc_pred=None,
    save_dir="./",
    skip_cadence=5,
    fps=60,
):

    if os.path.isdir(root_dir + "animation_images") != True:
        os.mkdir(root_dir + "animation_images")

    os.chdir(root_dir + "animation_images")

    fig, (ax0, ax1) = plt.subplots(1, 2, sharey=True, figsize=(8, 6))

    # panel 1
    im0 = ax0.imshow(data_stack_1[:, :, 0], origin="lower", vmin=vmin1, vmax=vmax1)
    time_text0 = ax0.text(1.0, 1.0, "", size=10, color="white")
    # frame_number0 = ax0.text(92.0, 120.0, '', size=10, color='white')
    frame_number0 = ax0.text(92.0, 240.0, "", size=10, color="white")

    divider0 = make_axes_locatable(ax0)
    cax0 = divider0.append_axes("right", size="2%", pad=0.1)
    cbar0 = fig.colorbar(im0, ax=ax0, cax=cax0, label="Relative flux (e/s)")

    # panel 2
    im1 = ax1.imshow(data_stack_2[:, :, 0], origin="lower", vmin=vmin2, vmax=vmax2)
    time_text1 = ax1.text(1.0, 1.0, "", size=10, color="white")
    # frame_number1 = ax1.text(65.0, 120.0, '', size=10, color='white')
    frame_number1 = ax1.text(65.0, 240.0, "", size=10, color="white")

    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="2%", pad=0.1)
    cbar1 = fig.colorbar(im1, ax=ax1, cax=cax1, label="Relative flux (e/s)")

    # set plot features
    ax0.set_ylim(0, len(data_stack_1))

    ax0.set_xticks([])
    ax0.set_yticks([])

    ax1.set_xticks([])
    ax1.set_yticks([])

    # ax0.set_xlabel('x pixel')
    # ax0.set_ylabel('y pixel')

    # ax1.set_xlabel('x pixel')
    # ax1.set_ylabel('y pixel')

    plt.tight_layout()

    # iteratively adjust figure

    for n in range(0, len(data_stack_1[0, 0])):

        if n % skip_cadence == 0:
            # set panel 1 at each frame
            im0.set_data(data_stack_1[:, :, n])
            # im0.set_clim(vmin0, vmax0)
            im0.set_clim(vmin1, vmax1)

            # set panel 2 at each frame
            im1.set_data(data_stack_2[:, :, n])
            if loc_pred != None:
                ax1.scatter(
                    loc_pred[n, 0],
                    loc_pred[n, 1],
                    edgecolors="orange",
                    s=mpl.rcParams["lines.markersize"] ** 3.0,
                    alpha=0.05,
                    facecolors="none",
                )
            im1.set_clim(vmin2, vmax2)

            # cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Relative flux (e/s)')
            t = Time(times[n], format="jd").iso
            time_text0.set_text(t)
            time_text1.set_text(t)

            frame_number0.set_text("Frame %i" % (n + 1))
            frame_number1.set_text(
                "%i/%i Summed Frames" % (n + 1, len(data_stack_1[0, 0]))
            )
            plt.savefig("animation_frame%04d.png" % (n), dpi=800)

    filenames_ordered = sorted(glob.glob("*.png"), key=number)
    # print(filenames_ordered)
    ims = [
        imageio.imread(f) for f in filenames_ordered
    ]  # glob.glob("*.png")]#sorted(glob.glob("*.png"), key=number)]
    imageio.mimwrite("animation_2panel.mp4", ims, fps=60)

    # with imageio.get_writer('animation_2panel.mp4', mode='I', fps=fps) as writer:
    #    for filename in filenames_ordered:
    #        image = imageio.imread(filename)
    #        writer.append_data(image)
    # os.system("ffmpeg -framerate 60 -pattern_type glob -i animation_frame%04d.png -c:v libx264 -pix_fmt yuv420 animation_2panel.mp4")

    plt.clf()
    plt.close()


def make_animation_lombscargle(times_all, fluxes_all, window_size, root_dir='./'):

    if os.path.isdir(root_dir + "Lomb-Scargle_animation") != True:
        os.mkdir(root_dir + "Lomb-Scargle_animation")

    os.chdir(root_dir + "Lomb-Scargle_animation")

    frequency = np.linspace(0, 2, 5000)

    min_window_midpoint = 0.5 * window_size
    max_window_midpoint = (len(times_all) - 0.5*window_size)

    for ind_obs in range(0, len(times_all)):

        start_window = int(ind_obs - min_window_midpoint)
        end_window = int(ind_obs + min_window_midpoint)

        # set start and end for moving window
        if ind_obs < min_window_midpoint:
            pass
        elif ind_obs >= max_window_midpoint:
            pass
        else:
            if ind_obs%10 == 0:
                fig, ax = plt.subplots(2, figsize=(10, 8))
                ax[0].scatter(times_all, fluxes_all, color='purple', s=10)
                #ax[0].axvline(times_all[start_window], alpha=0.4, color='gray')
                ax[0].axvspan(times_all[start_window], times_all[end_window], alpha=0.2, color='gray')
                ax[0].set_xlabel('time (JD)')
                ax[0].set_ylabel('summed flux (e/s)')
                ax[0].set_xlim(np.min(times_all), np.max(times_all))

                power_ls_window = LombScargle(times_all[start_window:end_window], fluxes_all[start_window:end_window]).power(frequency)
                ax[1].plot(frequency, power_ls_window, color='lightseagreen')
                #ax[1].axvline(24./19., label='Neptune rotation period', color='k')
                #ax[1].axvline(1./5.8769, label='Triton orbital period', color='r')
                ax[1].set_xlabel('frequency (cycles/day)')
                ax[1].set_ylabel('power')
                ax[1].set_xlim(np.min(frequency), np.max(frequency))
                ax[1].legend()
                plt.savefig('ls_%04d.png' %(ind_obs), dpi=400)
                plt.clf()
                plt.close()

    filenames_ordered = sorted(glob.glob("*.png"), key=number)
    ims = [imageio.imread(f) for f in filenames_ordered]#glob.glob("*.png")]#sorted(glob.glob("*.png"), key=number)]
    imageio.mimwrite('animation_ls.mp4', ims, fps=30)


"""
target_name = 'Neptune'
sector = 42
cam = 2
ccd = 4

root_dir = '/Users/malenarice/Desktop/Research/TESS_SolarSystem/%s/s%i_cam%i_ccd%i/' %(target_name, sector, cam, ccd)
fits_filename = root_dir + 'stacks_%s_s%i_cam%i_ccd%i.fits' %(target_name, sector, cam, ccd)

hdu = fits.open(fits_filename)
data_stack_bkgsub = hdu[4].data
times = hdu[6].data
#sortind = np.argsort(times)
#data_stack_bkgsub = data_stack_bkgsub[:,:,sortind]
#times = times[sortind]

make_animation_1panel(data_stack_bkgsub, times, save_dir=root_dir+'%s_frames_s%i_cam%i_ccd%i/' %(target_name, sector, cam, ccd))"""
