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
from fullsector import get_peak_frequencies
from subsectors import get_bin_edges, insert_gap_bin, expand_by_time_ranges

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
tableau_cb10 = plt.rcParams['axes.prop_cycle'].by_key()['color'] 

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

def plot_periodogram(axs, frequency, power, fap_stack, color, 
                     probabilities=[10, 1, 0.01], 
                     xlim=[8, 24], 
                     legend_loc="upper right", 
                     log=True, period_limit=None):
    x_mask = (24/frequency >= xlim[0]) & (24/frequency <= xlim[1])
    frequency = frequency[x_mask]
    power = power[x_mask]
    axs.plot(24 / frequency, power, color=color, linewidth=2)
    for i, fap in enumerate(np.flip(fap_stack)):
        fap = fap[x_mask]
        axs.plot(24 / frequency, fap, linestyle='dashed', label=f'{np.flip(probabilities)[i]}% FAP Line', linewidth=2)

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    axs.yaxis.set_major_formatter(formatter)
    axs.xaxis.set_minor_locator(AutoMinorLocator())
    axs.set_xlabel("Period [hours]")

    axs.set_xlim(xlim[0], xlim[1])
    axs.set_ylim(0, None)

    if log:
        axs.set_xscale('log')
        x_major = mpl.ticker.LogLocator(base=10.0, numticks=5)
        axs.xaxis.set_major_locator(x_major)
        x_minor = mpl.ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10)
        axs.xaxis.set_minor_locator(x_minor)
        # OPTIONAL: Only use NullFormatter if you really don't want labels
        # axs.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    if period_limit is not None:
        axs.axvline(period_limit, color=plot_colors_rgb[3], linestyle='dashdot', label=f'Minimum $P_{{max}}$', linewidth=2)

    axs.legend(fontsize=12, loc=legend_loc)
    axs.grid(True)
    axs.figure.canvas.draw()  # ← Move here, AFTER all formatting


def plot_lightcurve_and_periodogram(planet, 
                                    lc_list, 
                                    periodogram_list, 
                                    sector_list, 
                                    flux_string='raw', root=None, 
                                    xlim=[8,20], log=True, period_limits=[17.52, 18.72]):

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
                        lc[flux_string], 
                        color=color)

    for i, periodogram in enumerate(periodogram_list):
        if i < 2:
            periodogram_limit = period_limits[0]
        else:
            periodogram_limit = period_limits[1]
        plot_periodogram(axs[i + len(lc_list)], periodogram['frequency'], 
                            periodogram['power'], 
                            periodogram['false_alarm_levels'], 
                            color=color, probabilities=[10, 1, 0.01], 
                            xlim=xlim, log=log, period_limit=periodogram_limit)
        axs[i + len(lc_list)].set_xlim(xlim[0], xlim[1])

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

def plot_subsec_latsols(ax, subseclat, subsecstd, eqn, 
                        flip=False, fmt='o', 
                        color='xkcd:light blue',
                        alpha=0.5):
    for lat, std in zip(subseclat, subsecstd):
        std = np.reshape(std, (-1, 1))

        if flip:
            lat = -lat
        ax.errorbar(eqn(np.radians(lat)), 
                            lat, 
                            yerr=std, 
                            fmt=fmt, color=color,
                            capsize=3, alpha=alpha)

# def plot_mosaic_latitudes(mosaic_data, eqns, lats, stds, plot_colors, 
#                           title, planet, sector=None, vmin_percentile=None, vmax_percentile=None):
#     binary = sns.color_palette("viridis", as_cmap=True)
#     #plt.style.use('viridis')

#     phi = np.linspace(-np.pi/2, np.pi/2, 361)

#     fig = plt.figure(figsize=(10, 6))
#     if sector is not None:
#         fig.suptitle(f"Sector {sector}, {title}", fontsize=16, x=0.05, horizontalalignment='left')

#     gs = gridspec.GridSpec(2, 6, height_ratios=[1, 1])

#     ax = fig.add_subplot(gs[0:2, 0:6])
#     ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.05)

#     if planet == "Neptune":
#         # phi = np.linspace(-np.pi/2, 0, 181)
#         lat, stds = np.array(lats), np.array(stds)
#         plot_neptune_equations(ax, phi, plot_colors, h_band=False)
#         for i, equation in enumerate(eqns):
#             plot_lat_solutions(ax, lat[i], stds[i], equation, 42, plot_colors[i])

#         ax.set_ylim(-90, 90)
#         ax.set_xlim(-500, 400)

#     elif planet == "Uranus":
#         plot_uranus_equations(ax, phi, plot_colors)
#         markers = ["o", "^", "s"]
#         for j, lat in enumerate(lats):
#             for i, equation in enumerate(eqns):
#                 plot_lat_solutions(ax, lat[i], stds[j][i], equation, 42, plot_colors[i], even=False, marker=markers[j])

#         ax.set_ylim(-90,90)
#         ax.set_xlim(-100, 300)

#     ax.set_xlabel(r"Wind Speed $[ms^{-1}]$")
#     ax.xaxis.set_label_position('top') 

#     xlim, ylim = ax.get_xlim(), ax.get_ylim()

#     x1n = np.linspace(-180, 181, 30)

#     def forward(x):
#         return np.interp(x, x1n, np.linspace(xlim[0], xlim[1], 30))

#     def inverse(x):
#         return np.interp(x, np.linspace(xlim[0], xlim[1], 30), x1n)

#     if planet == "Neptune":
#         # mosaic_data = mosaic_data[180:360, :]
#         mosaic_data = mosaic_data/np.nanmean(mosaic_data)
#         im = ax.imshow(mosaic_data, extent=[xlim[0], xlim[1], ylim[0], ylim[1]], aspect='auto', cmap=binary, 
#                        vmin=np.percentile(mosaic_data, vmin_percentile), vmax=np.percentile(mosaic_data, vmax_percentile))
        
#         ax.legend(loc='center right')
#     elif planet == "Uranus":
#         # mosaic_data = mosaic_data[0:180, :]
#         mosaic_data = mosaic_data/np.nanmean(mosaic_data)
#         im = ax.imshow(mosaic_data, extent=[xlim[0], xlim[1], ylim[0], ylim[1]], aspect='auto', cmap=binary, 
#                        vmin=np.percentile(mosaic_data, vmin_percentile), vmax=np.percentile(mosaic_data, vmax_percentile))
#         ax.legend(loc='center right')

#     fig.colorbar(im, cax=cax, label=r"Normalized Flux $[e^{-}s^{-1}]$")
#     ax.set_ylabel(r"Latitude $[{^\circ}]$")

#     secax = ax.secondary_xaxis("bottom", functions=(inverse, forward))

#     secax.set_xticks(np.linspace(-180, 180, 13))  # longitude ticks
#     secax.set_xticklabels([f"{val:.0f}" for val in np.linspace(-180, 180, 13)])
#     secax.set_xlabel(r"Longitude $[{^\circ}]$")

#     if title:
#         fig.suptitle(title, fontsize=36/2)
#     # return all_bright_points

# def plot_summed_mosaics(summed_data, eqns, lats, stds, plot_colors, 
#                         planet, sector=None, save=False, log=True, clip_percentile=97, 
#                         gradient=False, vmin_percentile=None, vmax_percentile=None):
#     # summed_data = None
#     # for dir in directories:
#     #     for file in os.listdir(dir):
#     #         if file.endswith('.fits'):
#     #             #print(file)
#     #             hdul = fits.open(os.path.join(dir, file))
#     #             data = hdul[0].data
#     #             summed_data = data if summed_data is None else summed_data + data
#     #             hdul.close()

#     if gradient:
#         lat_gradient = np.abs(np.gradient(summed_data)[1])
#     else: 
#         lat_gradient = summed_data
        
#     #clip lat_gradient at 99th percentile for better visualization
#     lat_gradient_clipped = np.clip(lat_gradient, 0, np.percentile(lat_gradient, clip_percentile))

#     # set max values of lat_gradient to the mean 
#     lat_gradient_clipped[lat_gradient_clipped == np.percentile(lat_gradient, clip_percentile)] = np.median(lat_gradient)

#     title = None # (f"{planet} {sector}")
#     plot_mosaic_latitudes(lat_gradient_clipped, eqns, lats, stds, 
#                           plot_colors, planet=planet, 
#                           title=title, sector=None,vmin_percentile=vmin_percentile, vmax_percentile=vmax_percentile)
    
#     plt.tight_layout()
#     if save:
#         plt.savefig(f"{planet}_{sector}_opal.png", transparent=True, dpi=600)
#     else:
#         plt.show()

    # return bright_points

## subsector plots

def plot_split_lightcurves(axs, time_stack, flux_stack, title):
    axs.set_title(title)

    for i in range(len(time_stack)):
        axs.plot(time_stack[i] - 2457000, flux_stack[i], label=f"Subsector {i + 1}")
    axs.grid()

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))  # Forces scientific notation for small/large numbers
    axs.set_xlabel("Days [BTJD]")

    axs.grid(True)
    axs.yaxis.set_major_formatter(formatter)
    axs.xaxis.set_minor_locator(AutoMinorLocator())
    axs.figure.canvas.draw()


def plot_split_periodograms(axs, 
                            frequency, 
                            power_stack, 
                            fap_stack, xlim=[8, 24]):
    # axs.set_title(title)

    for i in range(len(power_stack)):
        axs.plot(24/frequency[i], power_stack[i], linewidth=2, label=f"Bin {i + 1}")
        axs.axhline(fap_stack[i], color=tableau_cb10[i], linestyle='--')
    axs.grid()

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))  # Forces scientific notation for small/large numbers

    axs.grid(True)
    #axs.yaxis.set_major_formatter(formatter)
    axs.xaxis.set_minor_locator(AutoMinorLocator())
    axs.figure.canvas.draw()
    axs.set_xlim(xlim[0], xlim[1])
    axs.set_xlabel("Period [hours]")

def plot_heatmap(axs, time_stack, frequency, max_frequency, power, fap_stack, min_per, max_per,
                 btjd_offset=2400, round_decimals=1,
                 gap_factor=3, vmin=0, vmax=1, output_table=False,
                 xlim=None):

    period_limit = 1/max_frequency   # in hours

    bin_starts, bin_ends = get_bin_edges(time_stack,
                                         btjd_offset=btjd_offset,
                                         round_decimals=round_decimals)
    
    # Insert a dummy bin if there's a large data gap
    bin_starts, bin_ends, gap_idx = insert_gap_bin(bin_starts, bin_ends,
                                                   gap_factor=gap_factor)
    insert_bins = np.insert(bin_starts, gap_idx, bin_ends[gap_idx - 1])
    bin_starts =  np.concatenate([insert_bins, [bin_ends[-1]]])
    new_power = []
    time_ranges = []
    tot_time = 0

    period_grid = np.linspace(min_per, max_per, 100)
    peak_freqs, periods = [], []
    for i, time in enumerate(time_stack):
        time = time - 2457000
        start_time = np.round(time[0], decimals=round_decimals + 1)
        end_time = np.round(time[-1], decimals=round_decimals + 1)
        range_time = np.round(time[-1] - time[0], decimals=round_decimals + 1)
        tot_time += range_time
        peak_frequencies, power_vals = get_peak_frequencies(frequency[i], 
                                                            power[i], 
                                                            np.full_like(frequency[i], 0))
        mask = peak_frequencies > max_frequency
        peaks_below_limit, power_below_limit = peak_frequencies[mask], power_vals[mask]

        # if len(peaks_below_limit) == 0:
        #     continue
        
        peak_freqs.append(peaks_below_limit[np.argmax(power_below_limit)])
        period = 24 / peaks_below_limit[np.argmax(power_below_limit)]
        periods.append(period)
        output_period = np.round(period, decimals=2)

        if output_table:
            if (np.max(power_below_limit) < fap_stack[i]) or (period > period_limit):
                print(f" & No Peak & {start_time} - {end_time} & {range_time} \\\\")
            else:
                print(f" & {output_period} & {start_time} - {end_time} & {range_time} \\\\")

        period_periods = np.flip(24/frequency[i])
        period_power = np.flip(np.array(power[i], dtype=float))

        # Find the valid period range in your native data
        min_period = period_periods.min()
        max_period = period_periods.max()

        # Create output array filled with NaN
        resamp_period_power = np.full_like(period_grid, np.nan, dtype=float)

        # Mask for valid period range
        valid_mask = (period_grid >= min_period) & (period_grid <= max_period)

        # Interpolate only within valid range
        resamp_period_power[valid_mask] = np.interp(period_grid[valid_mask], 
                                                    period_periods, 
                                                    period_power)
        
        new_power.append(resamp_period_power)
        time_ranges.append(range_time)

    print(f"TOTAL BIN TIME: {tot_time}")
    new_power = np.array(new_power)
    time_ranges = np.array(time_ranges)
    print(f"time ranges: {time_ranges}")

    print(time_ranges[gap_idx - 1], time_ranges[gap_idx])
    print(bin_starts[gap_idx - 1], bin_starts[gap_idx], bin_starts[gap_idx + 1])
    dummy_row = np.full_like(new_power[0], np.nan)  # NaN instead of value
    gap_power = np.insert(new_power, gap_idx, dummy_row, axis=0)

    dummy_gap = bin_starts[gap_idx] - bin_starts[gap_idx-1]
    gap_time_ranges = np.insert(time_ranges, gap_idx, dummy_gap)  # Insert a zero for the gap

    stretch_power, stretched_rows = expand_by_time_ranges(gap_power, gap_time_ranges, scale=20)
    print(gap_power.shape)
    print(stretch_power.shape)
    print(gap_time_ranges.shape)
    print(f"stretched_rows: {len(stretched_rows)}")

    # Create colormap that shows NaN as white
    cmap = plt.cm.coolwarm.copy()
    cmap.set_bad(color="white")

    axs.imshow(stretch_power, interpolation='nearest', aspect='auto',
                    origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)

    period_range = np.arange(xlim[0], xlim[1] + 1, 2)
    axs.set_yticks(np.linspace(0, 99, len(period_range)), period_range, fontsize=12)
    x_ticks = np.cumsum([0] + [s.shape[0] for s in stretched_rows])
    x_labels = [f"{t:.1f}" for t in bin_starts]

    print(f"x_ticks: {x_ticks}")
    print(f"x_labels: {x_labels}")

    axs.set_xlabel(f"Days [BTJD - {btjd_offset}]")
    # axs.set_title(title)
    axs.set_xticks(x_ticks)
    axs.set_xticklabels(x_labels, rotation=45, ha='right')

    # Add "Data gap" label
    if gap_idx is not None:
        axs.text(x_ticks[gap_idx] + 10, 50, "data gap", ha="center", va="center",
                 rotation=90, fontsize=18, color="black",
                 bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))

    return np.array(peak_freqs), np.array(periods), np.array(stretch_power), np.array(gap_time_ranges), stretched_rows

def plot_subsector_heatmap(planet, 
                           subtimes, subfluxes, 
                           subfreqs, subpower, subfap, 
                           labels, max_freq, offset, 
                           min_per, max_per, xlim=None):

    if xlim is None:
        xlim = [min_per, max_per]

    fig = plt.figure(figsize=(20, 18))
    gs = fig.add_gridspec(nrows=3, ncols=7, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1, 1, 1, 1, 0.1], wspace=0.3)

    if planet == "Neptune":

        # light curves
        ax1 = fig.add_subplot(gs[0, 0:3])
        ax2 = fig.add_subplot(gs[0, 3:6])
        ax1.set_ylabel(r"Flux [e$^{-}$s$^{-1}$]")
        lc_axs = [ax1, ax2]

        # periodograms
        ax3 = fig.add_subplot(gs[1, 0:3])
        ax4 = fig.add_subplot(gs[1, 3:6])
        ax3.set_ylabel("Power")
        pg_axs = [ax3, ax4]

        # heat maps
        ax5 = fig.add_subplot(gs[2, 0:3])
        ax5.set_ylabel("Period [hours]")
        ax6 = fig.add_subplot(gs[2, 3:6])

        hm_axs = [ax5, ax6]

        # colorbar
        ax7=fig.add_subplot(gs[2,6])
        ax7.set_ylabel("Power")
        #remove xaxis ticks
        ax7.set_xticks([])

        cbar_ax = ax7
    
    elif planet == "Uranus":
        # light curves
        ax1 = fig.add_subplot(gs[0, 0:2])
        ax1.set_ylabel(r"Flux [e$^{-}$s$^{-1}$]")
        ax2 = fig.add_subplot(gs[0, 2:4])
        ax3 = fig.add_subplot(gs[0, 4:6])

        lc_axs = [ax1, ax2, ax3]

        # periodograms
        ax4 = fig.add_subplot(gs[1, 0:2])
        ax4.set_ylabel("Power")
        ax5 = fig.add_subplot(gs[1, 2:4])
        ax6 = fig.add_subplot(gs[1, 4:6])

        pg_axs = [ax4, ax5, ax6]

        # heatmaps
        ax7 = fig.add_subplot(gs[2, 0:2])
        ax7.set_ylabel("Period [hours]")
        ax8 = fig.add_subplot(gs[2, 2:4])
        ax9 = fig.add_subplot(gs[2, 4:6])

        hm_axs = [ax7, ax8, ax9]
        
        # colorbar
        ax10=fig.add_subplot(gs[2,6])
        ax10.set_ylabel("Power")
        #remove xaxis ticks
        ax10.set_xticks([])

        cbar_ax = ax10

    for j, ax in enumerate(lc_axs):
        plot_split_lightcurves(ax, subtimes[j], subfluxes[j], labels[j])
    ax1.legend(fontsize=10, ncol=2)

    for j, ax in enumerate(pg_axs):
        plot_split_periodograms(ax, subfreqs[j], subpower[j], subfap[j], xlim=xlim)
    # heatmaps

    period_powers = []
    for j in range(len(hm_axs)):
        for i in range(len(subtimes[j])):
            pp = np.interp(np.linspace(min_per, max_per, 100), 
                           np.flip(np.asarray(24/subfreqs[j][i], dtype=float)), 
                           np.flip(np.asarray(subpower[j][i], dtype=float)))
            period_powers.append(pp)

    all_period_powers = np.array(period_powers)
    vmin = all_period_powers.min()
    vmax = all_period_powers.max()

    for j, ax in enumerate(hm_axs):
        plot_heatmap(ax, subtimes[j], subfreqs[j], 
                     max_freq, subpower[j], subfap[j], 
                     min_per, max_per, btjd_offset=offset[j], 
                     vmin=vmin, vmax=vmax, xlim=xlim)

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cb1 = mpl.colorbar.ColorbarBase(cbar_ax, cmap="coolwarm", norm=norm)

    plt.show()