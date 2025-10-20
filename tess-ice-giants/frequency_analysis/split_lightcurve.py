import numpy as np
from astropy.timeseries import LombScargle

from frequency_analysis.frequency_processing import linear_detrend

def split(x, m):
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

def get_segments_by_index(flux, segment_indices): 
    flux_segments = np.full_like(flux, np.nan)
    keep_indices = []

    for i in range(len(segment_indices)):
        start, end = segment_indices[i][0], segment_indices[i][1]

        if len(flux[start:end]) > len(flux)/10:
            keep_indices.append(i)
            #print(f"keep segment index {i}")
            flux_segments[start:end] = flux[start:end]

    return flux_segments, np.array(keep_indices)

def get_segments_by_time(time, flux, segment_indices, n_segments=10): 
    flux_segments = np.full_like(flux, np.nan)
    keep_indices = []

    time_per_segment = [(time[segment_indices[i][1] - 1] - time[segment_indices[i][0]]) for i, _ in enumerate(segment_indices)]
    total_time = sum(time_per_segment)

    for i in range(len(segment_indices)):
        start, end = segment_indices[i][0], segment_indices[i][1]

        if time_per_segment[i] > total_time/n_segments:
            keep_indices.append(i)
            #print(f"keep segment index {i}")
            flux_segments[start:end] = flux[start:end]

    return flux_segments, np.array(keep_indices)

def weight_segments_by_index(time, segment_indices, flux_segments, keep_indices):
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

def weight_segments_by_time(time, segment_indices, flux_segments, keep_indices):
    """Spits TESS sector into two main components & finds their proportion of the full light curve"""

    time_half, flux_half = [], []
    time_per_segment = [(time[segment_indices[i][1] - 1] - time[segment_indices[i][0]]) for i in keep_indices]
    total_time = sum(time_per_segment)
    print(time_per_segment)
    half_ratio = [t / total_time for t in time_per_segment]

    for i in keep_indices:
        #print(segment_indices[i])
        start, end = segment_indices[i][0], segment_indices[i][1]

        time_half.append(time[start:end])
        flux_half.append(flux_segments[start:end])

        # print(f"{seg_rat}")
        # print(f"{len(flux_segments[start:end])}")

    return time_half, flux_half, half_ratio

def stack_segments_by_index(time_half, flux_half, half_ratio, total_segs=10):

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


def stack_segments_by_time(time_half, flux_half, half_ratio, total_segs=10):
    time_stack, flux_stack = [], []

    for i, segment in enumerate(time_half):
        n = int(np.round(half_ratio[i] * total_segs))
        
        mask_nans = ~np.isnan(segment)
        segment = segment[mask_nans]
        time_length = (segment[-1] - segment[0]) / n
        print(time_length)

        for j in range(n):
            start = np.argmin(np.abs(segment - segment[0] - j * time_length))
            end = np.argmin(np.abs(segment  - segment[0] - (j + 1) * time_length))

            print(f"Segment {i} part {j}: {segment[start]} to {segment[end]}")
            this_time, this_segment = time_half[i][mask_nans][start:end], flux_half[i][mask_nans][start:end]

            time_stack.append(this_time)
            flux_stack.append(this_segment)

    return time_stack, flux_stack

def split_lightcurve(time, flux, m, total_segs=10, by_time=True):
    segment_indices = split(time, m)

    flux_segments, keep_indices = get_segments_by_time(time, flux, segment_indices)

    if by_time:
        time_half, flux_half, half_ratio = weight_segments_by_time(time, segment_indices, flux_segments, keep_indices)

        time_stack, flux_stack = stack_segments_by_time(time_half, flux_half, half_ratio, total_segs)

    else:
        time_half, flux_half, half_ratio = weight_segments_by_index(time, segment_indices, flux_segments, keep_indices)

        time_stack, flux_stack = stack_segments_by_index(time_half, flux_half, half_ratio, total_segs)

    return time_stack, flux_stack

def split_periodogram(time_stack, flux_stack, min_freq, max_freq, bootstrap=False, n_bootstrap=1000, fap_level=0.01):

    power_stack, fap_stack = [], []

    for i in range(len(time_stack)):
        frequency = np.linspace(min_freq, max_freq, 100)
        _, _, detrended = linear_detrend(time_stack[i], flux_stack[i])
        power = LombScargle(time_stack[i], detrended).power(frequency)

        if bootstrap:
            fap = LombScargle(time_stack[i], detrended).false_alarm_level(fap_level, method='bootstrap', method_kwds=dict(n_bootstraps=n_bootstrap))
        else: 
            fap = LombScargle(time_stack[i], detrended).false_alarm_level(fap_level)
        power_stack.append(power)
        fap_stack.append(fap)

    power_stack = np.array(power_stack)
    fap_stack = np.array(fap_stack)

    return frequency, power_stack, fap_stack

def split_data(times_list, flux_list, min_freq, max_freq, m=50, total_segs=10, by_time=True, bootstrap=False):

    all_times, all_flux, all_power, all_fap = [], [], [], []

    for i in range(len(times_list)): 

        time_stack, flux_stack = split_lightcurve(times_list[i], flux_list[i], m, total_segs, by_time=by_time)
        frequency, power_stack, fap_stack = split_periodogram(time_stack, flux_stack, min_freq, max_freq, bootstrap=bootstrap)

        all_times.append(time_stack)
        all_flux.append(flux_stack)
        all_power.append(power_stack)
        all_fap.append(fap_stack)
        frequency = frequency
        
    return all_times, all_flux, frequency, all_power, all_fap


# def split_data(file_list, target_id, observer_id, min_freq, max_freq, crop_list=[], m=50, total_segs=10):

#     all_times, all_flux, all_power = [], [], []

#     for i, data_file in enumerate(file_list): 

#         time, _, corrected_lightcurve = run_orbit_correction(target_id, observer_id, data_file, crop_list[i])

#         time_stack, flux_stack = split_lightcurve(time, corrected_lightcurve, total_segs, m=m)
#         frequency, power_stack = split_periodogram(time_stack, flux_stack, min_freq, max_freq)

#         all_times.append(time_stack)
#         all_flux.append(flux_stack)
#         all_power.append(power_stack)

#     return all_times, all_flux, frequency, all_power