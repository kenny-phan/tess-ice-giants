import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import seaborn as sns

from astropy.io import fits

from frequency_analysis.wind_equations import *

# ~~~ LATITUDE SOLUTIONS ~~~

def plot_lat_solutions(axs, latitudes, std, wind_eqn, sector, color, label=False, even=True, marker='o'):
    for i, latitude in enumerate(latitudes):
        if label:
            label = f"Sector {sector}" if i == 0 else None
        else: 
            label = None

        axs.errorbar(wind_eqn(np.radians(latitude)), latitude, yerr=std[i], fmt=marker, color=color, capsize=3, label=label)

        if even:
            axs.errorbar(wind_eqn(np.radians(latitude)), -latitude, yerr=std[i], fmt=marker, color=color, capsize=3)

def plot_neptune_equations(ax, phi, colors, h_band=False, linewidth=2):
    sromovsky1993_four = six_order_fit(-398, 1.88e-1, -1.2e-5)
    sromovsky1993_six = six_order_fit(-389, 1.53e-1, 1.01e-5, -3.1e-9)

    tollefson2013_kp = six_order_fit(-415, 2.35e-1, -2.23e-5)
    tollefson2014_kp = six_order_fit(-433, 2.4e-1, -2.73e-5)

    ax.plot(sromovsky1993_four(phi), phi * 180 / np.pi, label="Voyager 4th Order", color=colors[0], linewidth=linewidth)
    ax.plot(sromovsky1993_six(phi), phi * 180 / np.pi, label="Voyager 6th Order", color=colors[1], linewidth=linewidth)

    ax.plot(tollefson2013_kp(phi), phi * 180 / np.pi, label="Keck K' 2013", color=colors[2], linewidth=linewidth)
    ax.plot(tollefson2014_kp(phi), phi * 180 / np.pi, label="Keck K' 2014", color=colors[3], linewidth=linewidth)

    if h_band:
        tollefson2013_h = six_order_fit(-325, 1.58e-1, -1.21e-5)
        tollefson2014_h = six_order_fit(-292, 1.45e-1, -1.18e-5)
        ax.plot(tollefson2013_h(phi), phi * 180 / np.pi, label="T18 H 2013")
        ax.plot(tollefson2014_h(phi), phi * 180 / np.pi, label="T18 H 2014")

def plot_uranus_equations(ax, phi, colors, older=False, linewidth=2):

    ax.plot(sromovsky2012_odd_N(phi), phi * 180 / np.pi, label="Legendre 1997-2011", color=colors[0], linewidth=linewidth)
    s15 = make_sromovsky2015()
    ax.plot(s15(phi), phi * 180 / np.pi, label="Voyager & 2012-2014", color=colors[1], linewidth=linewidth)

    if older:
        tollefson2013_h = six_order_fit(-325, 1.58e-1, -1.21e-5)
        tollefson2014_h = six_order_fit(-292, 1.45e-1, -1.18e-5)
        ax.plot(tollefson2013_h(phi), phi * 180 / np.pi, label="T18 H 2013")
        ax.plot(tollefson2014_h(phi), phi * 180 / np.pi, label="T18 H 2014")

# ~~~ OPAL ~~~
def filter_chars(opal_file_name, range_start=39, range_end=43):
    return opal_file_name[range_start:range_end]


def plot_opal_data(directory, range_start=3, range_end=6, plot=False, 
                   labels=["F467M", "F547M", "F657N", "F763M", "F845M", "FQ619N", "FQ727N"]):

    file_list = os.listdir(directory)

    composite = np.zeros((361, 721))

    data_stack = []

    i = 0
    for file in sorted(file_list, key=filter_chars):

        if file.endswith(".fits"): 
            i += 1
            path = os.path.join(directory, file)

            with fits.open(path) as hdu:
                header = hdu[0].header
                data = hdu[0].data
                if plot:
                    print(filter_chars(file))
                    print(header)
                    plt.imshow(data)
                    plt.title(labels[i - 1])
                    plt.show()

                if i in range(range_start, range_end):
                    composite += data
                    data_stack.append(data)
                    if plot:
                        print(f"data max and min: {np.nanmax(data)}, {np.nanmin(data)}")
                        print(f"added {labels[i - 1]} to composite")
    if plot:
        plt.imshow(composite)
        plt.title("Composite of F657N, F763M, F845M")
        plt.show()

    return data_stack, composite

def select_bright_points(mosaic_data, axs, color, bins=60, percentile=95, plot=True, linewidth=2):
    threshold = np.percentile(mosaic_data, percentile)
    bright_points = np.argwhere(mosaic_data >= threshold)
    if plot:
        axs.hist(bright_points[:, 0], bins=bins, orientation="horizontal", density=True, label=f"{100 - percentile}%", histtype="step", color=color, linewidth=linewidth)
    return bright_points

def plot_mosaic_latitudes(mosaic_data, eqns, lats, stds, plot_colors, ibm, title, planet, sector=None, bins=np.arange(0, 361, 10)):
    binary = sns.color_palette("binary_r", as_cmap=True)

    phi = np.linspace(-np.pi/2, np.pi/2, 361)

    fig = plt.figure(figsize=(10, 6))
    if sector is not None:
        fig.suptitle(f"Sector {sector}, {title}", fontsize=16, x=0.05, horizontalalignment='left')

    gs = gridspec.GridSpec(2, 6, height_ratios=[1, 1])

    ax = fig.add_subplot(gs[0:2, 0:5])
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

    if planet == "Neptune":
        plot_neptune_equations(ax, phi, h_band=False)
        for i, equation in enumerate(eqns):
            plot_lat_solutions(ax, lats[i], stds[i], equation, 42, plot_colors[i])
    elif planet == "Uranus":
        plot_uranus_equations(ax, phi)
        for i, equation in enumerate(eqns):
            plot_lat_solutions(ax, lats[i], stds[i], equation, 42, plot_colors[i], even=False)
    ax.set_ylim(-90, 90)
    ax.set_xlabel(r"Wind Speed $[ms^{-1}]$")
    ax.xaxis.set_label_position('top') 

    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    x1n = np.linspace(-180, 181, 30)

    def forward(x):
        return np.interp(x, x1n, np.linspace(xlim[0], xlim[1], 30))

    def inverse(x):
        return np.interp(x, np.linspace(xlim[0], xlim[1], 30), x1n)

    ax.imshow(mosaic_data, extent=[xlim[0], xlim[1], ylim[0], ylim[1]], aspect='auto', cmap=binary)
    ax.set_ylabel(r"Latitude $[{^\circ}]$")

    secax = ax.secondary_xaxis("bottom", functions=(inverse, forward))

    secax.set_xticks(np.linspace(-180, 180, 13))  # longitude ticks
    secax.set_xticklabels([f"{val:.0f}" for val in np.linspace(-180, 180, 13)])
    secax.set_xlabel(r"Longitude $[{^\circ}]$")

    ax.legend(loc='center right')

    histax = fig.add_subplot(gs[0:2, 5:6])

    for i, percentile in enumerate([70, 80, 90, 95]):
        select_bright_points(mosaic_data, histax, color=ibm[-i], bins=bins, percentile=percentile, plot=True)

    histax.set_yticklabels([])
    histax.set_xticklabels([])
    histax.set_ylim(0, 360)

    histax.invert_yaxis()
    histax.grid()
    histax.legend()

    plt.show()

