import numpy as np
import numpy.polynomial.polynomial as poly 
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks

from frequency_analysis.orbit_correction import run_orbit_correction

def make_periodogram(time, flux, minimum_frequency=1, maximum_frequency=3, samples_per_peak=10, probabilities=[10, 1, 0.01], n_bootstrap=1000):
    ls = LombScargle(time, flux)

    frequency, power = ls.autopower(minimum_frequency=minimum_frequency, maximum_frequency=maximum_frequency, samples_per_peak=samples_per_peak) 
    
    false_alarm_levels = []

    for probability in probabilities:

        false_alarm = ls.false_alarm_level(probability/100., method='bootstrap', method_kwds=dict(n_bootstraps=n_bootstrap))
        false_alarm_array = np.full(frequency.shape, false_alarm)

        false_alarm_levels.append(false_alarm_array)

    false_alarm_levels = np.array(false_alarm_levels)
    
    return frequency, power, false_alarm_levels

def get_peak_frequencies(frequency, power, false_alarm_array):
    peaks, _ = find_peaks(power, height=false_alarm_array)
    return frequency[peaks], power[peaks]

def linear_detrend(time, flux):

    linear_fit = poly.polyfit(time, flux, 1)
    line = poly.polyval(time, linear_fit)

    detrended = flux - line + np.nanmean(flux)

    return linear_fit, line, detrended

def detrend_all(target_id, observer_id, data_file, crop_range=None):
    time, raw, orbit_corrected = run_orbit_correction(target_id, observer_id, data_file, crop_range=crop_range)
    _, _, detrended = linear_detrend(time, orbit_corrected)

    return time, raw, orbit_corrected, detrended

