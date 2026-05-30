import os 

import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.io import fits

from wind_equations import *

mpl.rcParams['mathtext.fontset'] = 'cm'          # Computer Modern serif
mpl.rcParams['mathtext.rm'] = 'serif'

plt.rcParams.update({'axes.linewidth' : 1.5, 
                     'ytick.major.width' : 1.5,
                     'ytick.minor.width' : 1.5,
                     'xtick.major.width' : 1.5,
                     'xtick.minor.width' : 1.5,
                     'xtick.labelsize': 12, 
                     'ytick.labelsize': 12,
                     'axes.labelsize': 18,
                     'axes.labelpad' : 5,
                     'axes.titlesize' : 24,
                     'axes.titlepad' : 10,
                     'font.family': 'Serif'
                    })
plt.style.use('tableau-colorblind10')

plot_colors_rgb = [
    (0/255, 107/255, 164/255),   # 006BA4
    (255/255, 128/255, 14/255),  # FF800E
    (171/255, 171/255, 171/255), # ABABAB
    (89/255, 89/255, 89/255),    # 595959
    (95/255, 158/255, 209/255),  # 5F9ED1
    (200/255, 82/255, 0/255),    # C85200
    (137/255, 137/255, 137/255), # 898989
    (162/255, 200/255, 236/255), # A2C8EC
    (255/255, 188/255, 121/255), # FFBC79
    (207/255, 207/255, 207/255)  # CFCFCF
]

## light curve and periodogram plotting functions
def scatter_data(axs, name, sector, x, y, color):
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))  # Forces scientific notation for small/large numbers
    axs.set_xlabel("Days [BTJD]")

    axs.set_title(name + " Sector " + str(sector))
    axs.scatter(x - 2457000, y, s=2, color=color)
    axs.grid(True)
    axs.yaxis.set_major_formatter(formatter)
    axs.xaxis.set_minor_locator(AutoMinorLocator())
    axs.figure.canvas.draw()

def plot_periodogram(axs, frequency, power, fap_stack, color, probabilities=[10, 1, 0.01], round_decimals=1):

    axs.plot(24 / frequency, power, color=color, linewidth=2)
    for i, fap in enumerate(np.flip(fap_stack)):
        axs.plot(24 / frequency, fap, linestyle='dashed', label=f'{np.flip(probabilities)[i]}% FAP Line', linewidth=2)

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))  # Forces scientific notation for small/large numbers

    axs.yaxis.set_major_formatter(formatter)
    axs.xaxis.set_minor_locator(AutoMinorLocator())
    axs.figure.canvas.draw()
    axs.set_xlabel("Period [hours]")


    axs.set_xlim(8, 24)
    axs.set_ylim(0, None)
    # axs.set_title(title)
    axs.legend(fontsize=12, loc="upper left")
    axs.grid(True)

def plot_lightcurve_and_periodogram(planet, lc_list, periodogram_list, sector_list, root=None):

    # Create the figure
    fig = plt.figure(figsize=(20, 12))        
    # Define a 2-row, 3-column grid
    gs = fig.add_gridspec(nrows=2, ncols=6, height_ratios=[1, 1])

    if planet == "Uranus":
        ax1 = fig.add_subplot(gs[0, 0:2])
        ax2 = fig.add_subplot(gs[0, 2:4])
        ax3 = fig.add_subplot(gs[0, 4:6])
        ax4 = fig.add_subplot(gs[1, 0:2])
        ax5 = fig.add_subplot(gs[1, 2:4])
        ax6 = fig.add_subplot(gs[1, 4:6])

        axs = [ax1, ax2, ax3, ax4, ax5, ax6]

        ax1.set_ylabel("Flux [e$^{-}$s$^{-1}$]")
        ax4.set_ylabel("Power")

        color="xkcd:sky blue"
    
    elif planet == "Neptune":
        ax1 = fig.add_subplot(gs[0, 0:3])
        ax2 = fig.add_subplot(gs[0, 3:6])
        ax3 = fig.add_subplot(gs[1, 0:3])
        ax4 = fig.add_subplot(gs[1, 3:6])

        axs = [ax1, ax2, ax3, ax4]

        ax1.set_ylabel("Flux [e$^{-}$s$^{-1}$]")
        ax3.set_ylabel("Power")

        color="xkcd:royal blue"

    for i, lc in enumerate(lc_list):
        scatter_data(axs[i], planet, 
                        sector_list[i], 
                        lc['time'], 
                        lc['orbit_corrected'], 
                        color=color)

    for i, periodogram in enumerate(periodogram_list):
        plot_periodogram(axs[i + len(lc_list)], periodogram['frequency'], 
                            periodogram['power'], 
                            periodogram['false_alarm_levels'], 
                            color=color, probabilities=[10, 1, 0.01])
        axs[i + len(lc_list)].set_xlim(8,20)

    plt.tight_layout()

    if root is not None:
        plt.savefig(root + f"{planet}.png", transparent=True, dpi=1000)
    else: 
        plt.show()

# opal maps plotting functions
def filter_chars(opal_file_name, range_start=39, range_end=43):
    return opal_file_name[range_start:range_end]

def plot_opal_data(directory, range_start=0, range_end=5, plot=False, 
                   labels=["F657N", "F763M", "F845M", "FQ619N", "FQ727N"]):

    file_list = os.listdir(directory)

    composite = np.zeros((361, 721))

    data_stack = []

    i = 0
    for file in sorted(file_list, key=filter_chars):

        if file.endswith(".fits"): 
            path = os.path.join(directory, file)

            with fits.open(path) as hdu:
                header = hdu[0].header
                data = hdu[0].data
                if plot:
                    # print(filter_chars(file))
                    # print(header)
                    plt.imshow(data)
                    plt.title(labels[i - 1])
                    plt.show()

                if i in range(range_start, range_end):
                    composite += data
                    data_stack.append(data)
                    if plot:
                        print(f"data max and min: {np.nanmax(data)}, {np.nanmin(data)}")
                        print(f"added {labels[i - 1]} to composite")

            i += 1

    if plot:
        plt.imshow(composite)
        plt.title("Composite of F657N, F763M, F845M")
        plt.show()

    return data_stack, composite

def normalize_by_num_filters(data_stack):
    # divide each pixel by the number of non zero summed pixels
    sum_stack = np.sum(data_stack, axis=0)
    div_array = np.zeros(data_stack.shape[1:])
    # print("div", div_array.shape)
    # print("data", np.sum(data_stack, axis=0).shape)

    for i in range(data_stack.shape[1]):
        for j in range(data_stack.shape[2]):
            div_array[i, j] = np.count_nonzero(data_stack[:, i, j])
            sum_stack[i, j] /= div_array[i, j] if div_array[i, j] > 0 else np.nan
    # data_stack = np.divide(data_stack, div_array)
    # set nans to zero
    sum_stack = np.nan_to_num(sum_stack)
    return sum_stack

def merge_pointings(dir1, dir2):
    data_stack_a, _ = plot_opal_data(dir1, plot=False)
    data_stack_b, _ = plot_opal_data(dir2, plot=False)
    data_stack = np.array(data_stack_a + data_stack_b)
    div_stack = normalize_by_num_filters(data_stack)
    return div_stack, data_stack

def take_latsol_with_largest_std(lat_array, std_array):
    for i in range(len(lat_array)): # for every wind equation
        for j in range(len(lat_array[i])): # for every solution
            if isinstance(lat_array[i][j], np.ndarray) and len(lat_array[i][j]) == 2:
                if std_array[i][j][0] > std_array[i][j][1]:
                    lat_array[i][j] = lat_array[i][j][0]
                    std_array[i][j] = std_array[i][j][0]
                else:
                    lat_array[i][j] = lat_array[i][j][1]
                    std_array[i][j] = std_array[i][j][1]

    return lat_array, std_array

def sort_ur_to_eqns(lat_arr, std_arr):
    new_lat_arr = np.vstack((np.concatenate(lat_arr[0:2]), np.concatenate(lat_arr[2:4])))
    new_std_arr = np.vstack((np.concatenate(std_arr[0:2]), np.concatenate(std_arr[2:4])))
    return new_lat_arr, new_std_arr

def plot_lat_solutions(axs, latitudes, std, wind_eqn, sector, color, label=False, even=True, marker='o'):
    for i, latitude in enumerate(latitudes):
        if label:
            label = f"Sector {sector}" if i == 0 else None
        else:
            label = None

        yerr = std[i]
        # If std[i] is a 2-element sequence, keep it as asymmetric yerr
        if isinstance(yerr, (list, tuple, np.ndarray)) and len(yerr) == 2:
            yerr = np.array(yerr)
        else:
            # scalar -> symmetric
            yerr = float(yerr)

        # print(wind_eqn(np.radians(latitude)).shape, latitude.shape, yerr.shape)

        # print(wind_eqn(np.radians(latitude)), latitude, yerr)
        yerr = np.reshape(yerr, (-1, 1))
        axs.errorbar(
            np.array(wind_eqn(np.radians(np.array(latitude)))),
            np.array(latitude),
            yerr=yerr,
            fmt=marker,
            color=color,
            capsize=3,
            label=label,
        )

        if even:
            # print(latitude)
            axs.errorbar(
                np.array(wind_eqn(np.radians(np.array(latitude)))),
                -np.array(latitude),
                yerr=yerr,
                fmt=marker,
                color=color,
                capsize=3,
            )

def make_rows_negative(lat_array, rows_to_negate):
    lat_arr_copy = lat_array.copy()
    for row in rows_to_negate:
        lat_arr_copy[row, :] = -lat_arr_copy[row, :]
    return lat_arr_copy

def plot_neptune_equations(ax, phi, colors, h_band=False, linewidth=2):
    sromovsky1993_four = six_order_fit(-398, 1.88e-1, -1.2e-5)
    sromovsky1993_six = six_order_fit(-389, 1.53e-1, 1.01e-5, -3.1e-9)

    tollefson2013_kp = six_order_fit(-415, 2.35e-1, -2.23e-5)
    tollefson2014_kp = six_order_fit(-433, 2.4e-1, -2.73e-5)

    ax.plot(sromovsky1993_four(phi), phi * 180 / np.pi, label="Sromovsky+ (1993) [4th Ord.]", color=colors[0], linewidth=linewidth)
    ax.plot(sromovsky1993_six(phi), phi * 180 / np.pi, label="Sromovsky+ (1993) [6th Ord.]", color=colors[1], linewidth=linewidth)

    ax.plot(tollefson2013_kp(phi), phi * 180 / np.pi, label="Tollefson+ (2018) [K' 2013]", color=colors[2], linewidth=linewidth)
    ax.plot(tollefson2014_kp(phi), phi * 180 / np.pi, label="Tollefson+ (2018) [K' 2014]", color=colors[3], linewidth=linewidth)

    if h_band:
        tollefson2013_h = six_order_fit(-325, 1.58e-1, -1.21e-5)
        tollefson2014_h = six_order_fit(-292, 1.45e-1, -1.18e-5)
        ax.plot(tollefson2013_h(phi), phi * 180 / np.pi, label="T18 H 2013")
        ax.plot(tollefson2014_h(phi), phi * 180 / np.pi, label="T18 H 2014")

def plot_uranus_equations(ax, phi, colors, linewidth=2):
    # Legendre 1997-2011, Voyager & 2012-2014
    ax.plot(sromovsky2012_odd_N(phi), phi * 180 / np.pi, label="Sromovsky+ (2012) [1997-2011]", color=colors[0], linewidth=linewidth)
    s15 = make_sromovsky2015()
    ax.plot(s15(phi), phi * 180 / np.pi, label="Sromovsky+ (2015)", color=colors[1], linewidth=linewidth)

def plot_mosaic_latitudes(mosaic_data, eqns, lats, stds, plot_colors, 
                          title, planet, sector=None, vmin_percentile=None, vmax_percentile=None):
    binary = sns.color_palette("viridis", as_cmap=True)
    #plt.style.use('viridis')

    phi = np.linspace(-np.pi/2, np.pi/2, 361)

    fig = plt.figure(figsize=(10, 6))
    if sector is not None:
        fig.suptitle(f"Sector {sector}, {title}", fontsize=16, x=0.05, horizontalalignment='left')

    gs = gridspec.GridSpec(2, 6, height_ratios=[1, 1])

    ax = fig.add_subplot(gs[0:2, 0:6])
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    if planet == "Neptune":
        # phi = np.linspace(-np.pi/2, 0, 181)
        lat, stds = np.array(lats), np.array(stds)
        plot_neptune_equations(ax, phi, plot_colors, h_band=False)
        for i, equation in enumerate(eqns):
            plot_lat_solutions(ax, lat[i], stds[i], equation, 42, plot_colors[i])

        ax.set_ylim(-90, 90)
        ax.set_xlim(-500, 400)

    elif planet == "Uranus":
        plot_uranus_equations(ax, phi, plot_colors)
        markers = ["o", "^", "s"]
        for j, lat in enumerate(lats):
            for i, equation in enumerate(eqns):
                plot_lat_solutions(ax, lat[i], stds[j][i], equation, 42, plot_colors[i], even=False, marker=markers[j])

        ax.set_ylim(-90,90)
        ax.set_xlim(-100, 300)

    ax.set_xlabel(r"Wind Speed $[ms^{-1}]$")
    ax.xaxis.set_label_position('top') 

    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    x1n = np.linspace(-180, 181, 30)

    def forward(x):
        return np.interp(x, x1n, np.linspace(xlim[0], xlim[1], 30))

    def inverse(x):
        return np.interp(x, np.linspace(xlim[0], xlim[1], 30), x1n)

    if planet == "Neptune":
        # mosaic_data = mosaic_data[180:360, :]
        mosaic_data = mosaic_data/np.nanmean(mosaic_data)
        im = ax.imshow(mosaic_data, extent=[xlim[0], xlim[1], ylim[0], ylim[1]], aspect='auto', cmap=binary, 
                       vmin=np.percentile(mosaic_data, vmin_percentile), vmax=np.percentile(mosaic_data, vmax_percentile))
        
        ax.legend(loc='center right')
    elif planet == "Uranus":
        # mosaic_data = mosaic_data[0:180, :]
        mosaic_data = mosaic_data/np.nanmean(mosaic_data)
        im = ax.imshow(mosaic_data, extent=[xlim[0], xlim[1], ylim[0], ylim[1]], aspect='auto', cmap=binary, 
                       vmin=np.percentile(mosaic_data, vmin_percentile), vmax=np.percentile(mosaic_data, vmax_percentile))
        ax.legend(loc='center right')

    fig.colorbar(im, cax=cax, label=r"Normalized Flux $[e^{-}s^{-1}]$")
    ax.set_ylabel(r"Latitude $[{^\circ}]$")

    secax = ax.secondary_xaxis("bottom", functions=(inverse, forward))

    secax.set_xticks(np.linspace(-180, 180, 13))  # longitude ticks
    secax.set_xticklabels([f"{val:.0f}" for val in np.linspace(-180, 180, 13)])
    secax.set_xlabel(r"Longitude $[{^\circ}]$")

    if title:
        fig.suptitle(title, fontsize=36/2)
    # return all_bright_points

def plot_summed_mosaics(summed_data, eqns, lats, stds, plot_colors, 
                        planet, sector=None, save=False, log=True, clip_percentile=97, 
                        gradient=False, vmin_percentile=None, vmax_percentile=None):
    # summed_data = None
    # for dir in directories:
    #     for file in os.listdir(dir):
    #         if file.endswith('.fits'):
    #             #print(file)
    #             hdul = fits.open(os.path.join(dir, file))
    #             data = hdul[0].data
    #             summed_data = data if summed_data is None else summed_data + data
    #             hdul.close()

    if gradient:
        lat_gradient = np.abs(np.gradient(summed_data)[1])
    else: 
        lat_gradient = summed_data
        
    #clip lat_gradient at 99th percentile for better visualization
    lat_gradient_clipped = np.clip(lat_gradient, 0, np.percentile(lat_gradient, clip_percentile))

    # set max values of lat_gradient to the mean 
    lat_gradient_clipped[lat_gradient_clipped == np.percentile(lat_gradient, clip_percentile)] = np.median(lat_gradient)

    title = None # (f"{planet} {sector}")
    plot_mosaic_latitudes(lat_gradient_clipped, eqns, lats, stds, 
                          plot_colors, planet=planet, 
                          title=title, sector=None,vmin_percentile=vmin_percentile, vmax_percentile=vmax_percentile)
    
    plt.tight_layout()
    if save:
        plt.savefig(f"{planet}_{sector}_opal.png", transparent=True, dpi=600)
    else:
        plt.show()

    # return bright_points