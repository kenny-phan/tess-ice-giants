import numpy as np
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks
from tqdm import tqdm

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

def save_periodograms(sector_data_list, sector_data_strings, root, flux_type='detrended', 
                      minimum_frequency=1, maximum_frequency=3,
                      samples_per_peak=10):
    
    for i, sector_data in tqdm(enumerate(sector_data_list)):
        frequency, power, false_alarm_levels = make_periodogram(sector_data['time'], sector_data[flux_type],
                                                                minimum_frequency=minimum_frequency, 
                                                                maximum_frequency=maximum_frequency,
                                                                samples_per_peak=samples_per_peak)
        peaks, peak_pows = get_peak_frequencies(frequency, power, false_alarm_levels[0])
        # print(false_alarm_levels)
        np.savez(root + f'{sector_data_strings[i]}_periodogram.npz', 
                frequency=frequency, power=power, false_alarm_levels=false_alarm_levels,
                peaks=peaks, peak_pows=peak_pows)
        
    
def group_and_average(arr1, arr2, mean=True):
    """
    Groups elements of arr1 so that the number of groups matches the size of arr2.
    Returns an array of the averages of each group.
    """
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    
    len1 = len(arr1)
    len2 = len(arr2)

    if len2 == 0:
        raise ValueError("arr2 must have non-zero length.")
    if len1 < len2:
        raise ValueError("arr1 must be at least as long as arr2.")
    
    # Compute group size (may not be perfect division)
    group_size = len1 / len2

    result = []
    for i in range(len2):
        start = int(round(i * group_size))
        end = int(round((i + 1) * group_size))
        group = arr1[start:end]
        if mean:
            avg = np.mean(group) if len(group) > 0 else 0
        else: 
            avg = np.median(group) if len(group) > 0 else 0
        result.append(avg)

    print(f"{len(arr1) - len(result)*group_size} data points discarded")
    return np.array(result)

    

