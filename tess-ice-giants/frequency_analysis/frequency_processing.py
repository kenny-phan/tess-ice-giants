import numpy as np
import numpy.polynomial.polynomial as poly 
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks

def make_periodogram(time, flux, minimum_frequency, maximum_frequency, samples_per_peak, probability, n_bootstrap=1000):
    ls = LombScargle(time, flux)

    frequency, power = ls.autopower(minimum_frequency=minimum_frequency, maximum_frequency=maximum_frequency, samples_per_peak=samples_per_peak) 

    false_alarm = ls.false_alarm_level(probability/100., method='bootstrap', method_kwds=dict(n_bootstraps=n_bootstrap))

    false_alarm_array = np.full(frequency.shape, false_alarm)
    
    return frequency, power, false_alarm_array

def get_peak_frequencies(frequency, power, false_alarm_array):
    peaks, _ = find_peaks(power, height=false_alarm_array)
    return frequency[peaks]

def linear_detrend(time, flux):

    linear_fit = poly.polyfit(time, flux, 1)
    line = poly.polyval(time, linear_fit)

    detrended = flux - line + np.mean(flux)

    return linear_fit, line, detrended