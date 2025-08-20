import numpy as np
from astropy.timeseries import LombScargle

from frequency_analysis.orbit_correction import run_orbit_correction

def split(x, m=100):
    """Get indices of light curve cut off points"""

    valid_x = x[~np.isnan(x)]
    gap_threshold = m * np.median(np.diff(valid_x))
    gaps = np.where(np.diff(x) > gap_threshold)[0]

    section_edges = np.concatenate(([0], gaps + 1, [len(x)]))

    segment_indices = []

    for i in range(len(section_edges) - 1):
        start, end = section_edges[i], section_edges[i + 1]
        segment_indices.append((start, end))

    return segment_indices

def get_flux_segments(flux, segment_indices): 
    flux_segments = np.full_like(flux, np.nan)
    keep_indices = []

    for i in range(len(segment_indices)):
        start, end = segment_indices[i][0], segment_indices[i][1]

        if len(flux[start:end]) > len(flux)/10:
            keep_indices.append(i)
            #print(f"keep segment index {i}")
            flux_segments[start:end] = flux[start:end]

    return flux_segments, np.array(keep_indices)


def weight_segments(time, segment_indices, flux_segments, keep_indices):
    """Spits TESS sector into two main components & finds their proportion of the full light curve"""

    time_half, flux_half, half_ratio = [], [], []

    for i in keep_indices:
        #print(segment_indices[i])
        start, end = segment_indices[i][0], segment_indices[i][1]
        seg_rat = len(flux_segments[start:end]) / len(flux_segments[~np.isnan(flux_segments)])

        time_half.append(time[start:end])
        flux_half.append(flux_segments[start:end])
        half_ratio.append(seg_rat)

        # print(f"{seg_rat}")
        # print(f"{len(flux_segments[start:end])}")

    return time_half, flux_half, half_ratio

def stack_segments(time_half, flux_half, half_ratio, total_segs=10):

    """some data loss up to n pixels"""

    time_stack, flux_stack = [], []

    for i, segment in enumerate(flux_half):
        n = int(np.round(half_ratio[i] * total_segs))

        index_length = len(segment[~np.isnan(segment)]) // n
        mask_nans = ~np.isnan(segment)

        #print(time_segments[i][mask_nans].shape, segment[mask_nans].shape)

        for j in range(n):
            start = j * index_length
            end = (j + 1) * index_length

            this_time, this_segment = time_half[i][mask_nans][start:end], segment[mask_nans][start:end]

            time_stack.append(this_time)
            flux_stack.append(this_segment)

            #print(time_half[i][mask_nans][start:end].shape, segment[mask_nans][start:end].shape)

        #print(len(segment[~np.isnan(segment)]))

    return time_stack, flux_stack

def split_lightcurve(time, flux, total_segs=10):
    segment_indices = split(time)

    flux_segments, keep_indices = get_flux_segments(flux, segment_indices)

    time_half, flux_half, half_ratio = weight_segments(time, segment_indices, flux_segments, keep_indices)

    time_stack, flux_stack = stack_segments(time_half, flux_half, half_ratio, total_segs)

    return time_stack, flux_stack


def split_periodogram(time_stack, flux_stack, min_freq, max_freq):

    power_stack = []

    for i in range(len(time_stack)):
        frequency = np.linspace(min_freq, max_freq, 100)
        power = LombScargle(time_stack[i], flux_stack[i]).power(frequency)

        power_stack.append(power)

    power_stack = np.array(power_stack)

    return frequency, power_stack


def split_data(file_list, target_id, observer_id, min_freq, max_freq, total_segs=10):

    all_times, all_flux, all_power = [], [], []

    for data_file in file_list: 

        time, _, corrected_lightcurve = run_orbit_correction(target_id, observer_id, data_file)

        time_stack, flux_stack = split_lightcurve(time, corrected_lightcurve, total_segs)
        frequency, power_stack = split_periodogram(time_stack, flux_stack, min_freq, max_freq)

        all_times.append(time_stack)
        all_flux.append(flux_stack)
        all_power.append(power_stack)

    return all_times, all_flux, frequency, all_power