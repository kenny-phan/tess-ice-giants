import numpy as np
from astropy.timeseries import LombScargle

from orbit_correction import linear_detrend
from fullsector import get_peak_frequencies,debug_print

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
    # print(time_per_segment)
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

    # Allocate segments with rounding
    n_list = [int(np.round(r * total_segs)) for r in half_ratio]
    remainder = total_segs - sum(n_list)
    
    # Add remainder to the segment with the largest ratio
    largest_idx = np.argmax(half_ratio)
    n_list[largest_idx] += remainder

    for i, segment in enumerate(time_half):
        n = n_list[i]
                
        mask_nans = ~np.isnan(segment)
        segment = segment[mask_nans]
        time_length = (segment[-1] - segment[0]) / n
        # print(time_length)

        for j in range(n):
            start = np.argmin(np.abs(segment - segment[0] - j * time_length))
            end = np.argmin(np.abs(segment  - segment[0] - (j + 1) * time_length))

            # print(f"Segment {i + 1} part {j + 1}: {segment[start]} to {segment[end]}")
            this_time, this_segment = time_half[i][start:end], flux_half[i][start:end]
            # print(this_time.shape, this_segment.shape)
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


def split_periodogram(time_stack, flux_stack, 
                      max_freq, freq_array_size=1000,
                      bootstrap=False, n_bootstrap=1000, 
                      fap_level=0.01):

    freq_stack, power_stack, fap_stack, peak_stack = [], [], [], []

    for i in range(len(time_stack)):
        baseline = time_stack[i][-1] - time_stack[i][0]
        min_freq = 1/(baseline/2)
        frequency = np.linspace(min_freq, max_freq, freq_array_size)
        _, _, detrended = linear_detrend(time_stack[i], flux_stack[i])
        power = LombScargle(time_stack[i], detrended).power(frequency)

        if bootstrap:
            fap = LombScargle(time_stack[i], detrended).false_alarm_level(fap_level, 
                                                                          method='bootstrap', 
                                                                          method_kwds=dict(n_bootstraps=n_bootstrap))
        else: 
            fap = LombScargle(time_stack[i], detrended).false_alarm_level(fap_level)

        peaks, _ = get_peak_frequencies(frequency, power, [fap])

        if len(peaks) > 0:
            peak_stack.append(peaks)

        freq_stack.append(frequency)
        power_stack.append(power)
        fap_stack.append(fap)
        
    power_stack = np.array(power_stack)
    fap_stack = np.array(fap_stack)
    freq_stack = np.array(freq_stack)
    return freq_stack, power_stack, fap_stack, peak_stack #std_stack

def split_data(times_list, flux_list, max_freq_arr, freq_array_size=500,
               m=50, total_segs=10, by_time=True, 
               bootstrap=False, 
               verbose=False, fap_level=0.01):

    all_times, all_flux, all_freq, all_power, all_fap, all_peak = [], [], [], [], [], []

    for i in range(len(times_list)): 
        debug_print(verbose, f"Processing dataset {i}")
        time_stack, flux_stack = split_lightcurve(times_list[i], flux_list[i], m, total_segs, by_time=by_time)
        frequency, power_stack, fap_stack, peak_stack = split_periodogram(time_stack, flux_stack, max_freq_arr[i], 
                                                                                     freq_array_size=freq_array_size,
                                                                                     bootstrap=bootstrap, fap_level=fap_level)

        all_times.append(time_stack)
        all_flux.append(flux_stack)
        all_power.append(power_stack)
        all_fap.append(fap_stack)
        all_peak.append(peak_stack)
        all_freq.append(frequency)
        
    return all_times, all_flux, all_freq, all_power, all_fap, all_peak

def get_bin_edges(time_stack, btjd_offset=2400, round_decimals=1):
    bin_edges = []
    for time in time_stack:
        # Use the first and last time of each bin
        bin_edges.append((np.round(time[0] - 2457000 - btjd_offset, round_decimals),
                          np.round(time[-1] - 2457000 - btjd_offset, round_decimals)))
        
    bin_starts = [edge[0] for edge in bin_edges]
    bin_ends = [edge[1] for edge in bin_edges]
    return bin_starts, bin_ends

def insert_gap_bin(bin_starts, bin_ends, gap_factor=3):
    """
    Insert a dummy bin into the bin edges if there's a large gap.
    
    Parameters
    ----------
    bin_starts, bin_ends : list
        Start and end times of bins.
    gap_factor : float
        Factor above median gap size to qualify as 'large'.
    
    Returns
    -------
    new_starts, new_ends : list
        Modified bin edges including a gap bin.
    gap_index : int or None
        Index where gap was inserted, None if no gap.
    """
    gaps = [bin_starts[i+1] - bin_ends[i] for i in range(len(bin_starts)-1)]
    if len(gaps) == 0:
        return bin_starts, bin_ends, None
    
    median_gap = np.median(gaps)
    large_gap_idx = np.argmax(gaps)  # largest gap
    if gaps[large_gap_idx] > gap_factor * median_gap:
        # Place dummy bin at midpoint of large gap
        midpoint = (bin_ends[large_gap_idx]) 
        new_starts = bin_starts[:large_gap_idx+1] + bin_starts[large_gap_idx+1:]
        new_ends   = bin_ends[:large_gap_idx+1]  + bin_ends[large_gap_idx+1:]
        return new_starts, new_ends, large_gap_idx+1
    else:
        return bin_starts, bin_ends, None

def expand_by_time_ranges(new_power, time_ranges, scale=10):
    """
    Expand each time bin in new_power horizontally proportional to its time span.

    Parameters
    ----------
    new_power : 2D array, shape (N_bins, N_periods)
        Power values per time bin.
    time_ranges : 1D array, shape (N_bins,)
        Duration (or relative width) of each bin.
    scale : float
        Controls total width of output array; higher = finer scaling.

    Returns
    -------
    expanded_power : 2D array
        Time-stretched array suitable for plt.imshow().
    """
    # Normalize to mean width = 1, then scale up for better resolution
    norm = np.mean(time_ranges)
    n_cols = np.round((time_ranges / norm) * scale).astype(int)
    # n_cols[n_cols < 1] = 1  # ensure at least one column per bin

    stretched_rows = [np.repeat(row[np.newaxis, :], n, axis=0) for row, n in zip(new_power, n_cols)]
    expanded_power = np.vstack(stretched_rows)
    return expanded_power.T, stretched_rows  # transpose so shape = (N_periods, total_time_pixels)


def sort_lat_std(latitudes, standard_deviations):
    sorted_latitudes = np.empty_like(latitudes, dtype=object)
    sorted_standard_deviations = np.empty_like(standard_deviations, dtype=object)
    for j in range(latitudes.shape[0]):
        lat_arr = latitudes[j, :]
        std_arr = standard_deviations[j, :]
        sorted_lat_arr = np.empty_like(lat_arr)
        sorted_std_arr = np.empty_like(std_arr)

        for k, (sublat, substd) in enumerate(zip(lat_arr, std_arr)):
            sorted_indices = [np.argsort(sublat[i]) for i in range(len(sublat))]
            sublat_sorted = [sublat[i][sorted_indices[i]] for i in range(len(sublat))]
            substd_sorted = [substd[i][sorted_indices[i]] for i in range(len(substd))]
            sorted_lat_arr[k] = sublat_sorted
            sorted_std_arr[k] = substd_sorted

        sorted_latitudes[j, :] = sorted_lat_arr
        sorted_standard_deviations[j, :] = sorted_std_arr

    return sorted_latitudes, sorted_standard_deviations


def lat_std_per_eqn(sorted_latitudes, sorted_standard_deviations, eqnidx, secidx):
    slat = sorted_latitudes[secidx, :]
    sstd = sorted_standard_deviations[secidx, :]
    thissec_latitudes, thissec_stds = [], []
    for lat, std in zip(slat, sstd):
        if lat[eqnidx] is not None and len(lat[eqnidx]) > 0:
            thissec_latitudes.append(lat[eqnidx][0])
            thissec_stds.append(std[eqnidx][0])

    return np.array(thissec_latitudes), thissec_stds


