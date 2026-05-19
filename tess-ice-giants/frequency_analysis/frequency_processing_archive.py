import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks
import os
from NWelch.src import TimeSeries


def plot_welchs_power_spectrum(time, flux, root_dir):

    ts = TimeSeries(time, flux)
    nyquist = (1./(2.*(np.median(np.diff(times_vetted)))))
    ts.frequency_grid(nyquist, oversample=4)
    ts.pow_FT()
    ts.powplot(yscale='linear', Welch=True)
    plt.tight_layout()
    plt.savefig(root_dir+'welchs_power_spectrum.png', dpi=300)
    plt.clf()
    plt.close()


def openPlot(x_label, y_label, title):
    plt.figure(figsize=(10, 6))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)


def endPlot(title, root_dir):
    plt.legend()
    plt.grid()
    plt.show()
    # title = title.replace(" ", "_")
    # plt.savefig(root_dir+title+".png", dpi=300)
    # plt.clf()
    # plt.close()

def plotPeriodogram(target_name, frequency, power, x_label, y_label, title, root_dir, periodicities=True, false_alarm=None, probability=None):
    #plot the periodogram
    openPlot(x_label, y_label, title)
    plt.plot(frequency, power, color="xkcd:royal blue")
    if periodicities:
        knownPeriodicities(target_name)

    if false_alarm is not None and probability is not None:
        false_alarm = np.atleast_1d(false_alarm)  # Ensure false_alarm is an array
        probability = np.atleast_1d(probability)
        if len(false_alarm) > 0 and len(probability) > 0:
            for index, level in enumerate(false_alarm):
                plt.axhline(level, alpha=0.5, color='grey', linestyle='dashed', label=f"Present if random: {probability[index]*100}%")
            # e.g. "we will observe a peak this high or higher approximately 0.4% of the time"

    plt.xlim(np.min(frequency), np.max(frequency))

    endPlot(title, root_dir)
    
def plotFittedSin(ls, time, flux, data_label, best_frequency, x_label, y_label, title, root_dir): #y_min, y_max,
    openPlot(x_label, y_label, title)
    best_period = 1 / best_frequency
    plt.plot(time, flux, 'ko', label=data_label, color="black", ms=0.5)
    plt.plot(time, ls.model(time, best_frequency), color="xkcd:royal blue", label=f'Best fit (P = {best_period:.2f} days)')

    #plt.ylim(y_min, y_max)
    endPlot(title, root_dir)

def plotDetrendedData(time, flux, data_label, flux_detrended, detrended_label, x_label, y_label, title, root_dir):
    # Plotting the original and detrended data
    openPlot(x_label, y_label, title)
    plt.scatter(time, flux, label=data_label, color="xkcd:royal blue", s=5)
    plt.plot(time, flux_detrended, color='r', label=detrended_label)
    endPlot(title, root_dir)

def knownPeriodicities(target_name):
    if target_name == "Neptune":
        #fill with related periodicities
        plt.axvline(1/13.7, alpha=0.5, color='black', label='TESS Orbit') #https://tess.mit.edu/science/
        plt.axvline(1/5.876854, alpha=0.5, color='r', label='Triton Orbit') #https://nssdc.gsfc.nasa.gov/planetary/factsheet/neptuniansatfact.html
        plt.axvline(24/18.734167401, alpha=0.5, color='darkorange', label='Equatorial Streamline') # Karkoschka, 2011 (400 m/s w wind)
        plt.axvline(1.4, alpha=0.5, color='cyan', label='Scooter Rotation') # Karkoschka, 2011 (17.14h period)
        plt.axvline(1.50316604348, alpha=0.5, color='purple', label='Neptune Rotation') # Karkoschka, 2011, based on Net South Polar Feature
    if target_name == "Uranus":
        plt.axvline(1/13.7, alpha=0.5, color='black', label='TESS Orbit') #https://tess.mit.edu/science/
        plt.axvline(24/17.24, alpha=0.5, color='purple', label='Uranus Rotation') # Karkoschka, 2011, based on Net South Polar Feature
        
def makePeriodogram(ls, minimum_frequency, maximum_frequency, samples_per_peak, probability):
    frequency, power = ls.autopower(minimum_frequency=minimum_frequency, maximum_frequency=maximum_frequency, samples_per_peak=samples_per_peak) #0.0001, 3

    n_bootstrap = 10000 # also try 100000 if it's fast
    false_alarm = ls.false_alarm_level(probability/100., method='bootstrap', method_kwds=dict(n_bootstraps=n_bootstrap))
    return frequency, power, false_alarm

def find_frequencies_ls(times, flux, target_name, sector, cam, ccd, root_dir, probability=0.00001, \
                        minimum_frequency=0.01, maximum_frequency=3, samples_per_peak=10, extra_data=False):
    print("Now processing periodogram...")

    ls = LombScargle(times, flux)

    frequency, power, false_alarm = makePeriodogram(ls, minimum_frequency, maximum_frequency, samples_per_peak, probability)

    false_alarm_array = np.full_like(power, false_alarm)

    peaks, _ = find_peaks(power, height=false_alarm_array)
    peak_frequencies = frequency[peaks]
    print("Peak frequencies:", peak_frequencies)

    peak_periods = 1 / peak_frequencies

    print("Peak periods:", peak_periods)

    tess_index = np.abs(peak_periods - 13.7).argmin()
    tess_period = peak_periods[tess_index]
    tess_frequency = peak_frequencies[tess_index]

    flux_mean = np.mean(flux)
    subtraction = ls.model(times, tess_frequency)
    flux_detrended = flux - subtraction + flux_mean

    ls_detrended = LombScargle(times, flux_detrended)
    frequency_dt, power_dt, false_alarm_dt = makePeriodogram(ls_detrended, minimum_frequency, maximum_frequency, samples_per_peak, probability)

    best_frequency_dt = frequency_dt[np.argmax(power_dt)]

    best_period_dt = 1 / best_frequency_dt

    #plots
    plotPeriodogram(target_name, frequency, power, 'Frequency (1/day)', 'Power', f'Lomb-Scargle Periodogram for {target_name}, Sector {sector}', root_dir, periodicities=True, false_alarm=false_alarm, probability=probability)

    if extra_data:
        plotFittedSin(ls, times, flux, 'Data', tess_frequency, 'Time (BTJD)', 'Flux', 'Data and Best-Fit Sine Curve', root_dir)
        plotDetrendedData(times, flux,'Original Data', flux_detrended, 'Detrended Data', 'Time (BTJD)', 'Flux', 'Detrended Data with Sinusoidal Model', root_dir)
        plotPeriodogram(target_name, frequency_dt, power_dt, 'Detrended Frequency (1/day)', 'Power', f'Detrended Lomb-Scargle Periodogram for {target_name}, Sector {sector}', root_dir, True, false_alarm=false_alarm_dt, probability=probability)

        np.save(root_dir + "peaks_frequencies_%s_s%i_cam%i_ccd%i.npy" % (target_name, sector, cam, ccd), peak_frequencies)
        np.save(root_dir + "false_alarm_%i_%s_s%i_cam%i_ccd%i.npy" % (probability, target_name, sector, cam, ccd), false_alarm)
        np.savetxt(
            root_dir + "periodogram_%s_s%i_cam%i_ccd%i.txt" % (target_name, sector, cam, ccd),
            np.vstack((frequency, power)),
    )

    return frequency, power, false_alarm

def load_lines(file_name):
    lines = np.loadtxt(file_name, comments="#", delimiter=" ", unpack=False)
    times = lines[0]
    flux = lines[1]
    return times, flux

'''
#-------UPDATE / example---------#
#where the diagnostic periodograms will save
target_name = "Neptune"
sector = 70
cam = 1
ccd = 1
buffer_pixel_size = 40
percent_pix_scatteredlight = 10

root_dir = '/scratch11/ktp9/TESS_SolarSystem/%s/s%i_cam%i_ccd%i/results/bps_%i_pps_%i/' %(target_name, sector, cam, ccd, buffer_pixel_size, percent_pix_scatteredlight)
file = root_dir + "lc_Neptune_s70_cam1_ccd1.txt"
times, flux = load_lines(file)
frequency, power, false_alarm = frequencyProcessing(times, flux, target_name, sector, cam, ccd, root_dir, probability=0.001,
                    minimum_frequency=0.0001, maximum_frequency=3, samples_per_peak=10, times_in_jd=False, extra_data=True)'''
#---------------------#
