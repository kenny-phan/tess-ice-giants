import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import AutoMinorLocator
import matplotlib as mpl

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
