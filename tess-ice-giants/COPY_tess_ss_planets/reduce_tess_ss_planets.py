from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from photutils.aperture import CircularAperture
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from photutils.aperture import aperture_photometry
import scipy.signal
import cv2
import os

from collections import deque
from bisect import insort, bisect_left
from itertools import islice
from astroquery.jplhorizons import Horizons
from scipy import optimize
from tess_stars2px import tess_stars2px_function_entry
from astroquery.mast import Tesscut
from astropy.utils.data import download_file
from astroquery.jplhorizons import Horizons
import lightkurve as lk  # imported only to use btjd as a time format. has to be before astropy.time
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astrocut import CutoutFactory

#from tess_solarsystem_planets.diagnostic_images import *
#from tess_solarsystem_planets.make_animations import *
#from tess_solarsystem_planets.frequency_processing import *
from diagnostic_images import *
from make_animations import *
from frequency_processing import *

# older, simpler versions; not curently in use  ------------------------------------


def subtract_median_image(data_stack, median_image):

    tiled_median_image = np.repeat(
        median_image[:, :, np.newaxis], len(data_stack[0, 0]), axis=2
    )
    data_stack_sub = data_stack - tiled_median_image

    return data_stack_sub


def running_median_insort(seq, window_size):

    seq = iter(seq)
    d = deque()
    s = []
    result = []

    for item in islice(seq, window_size):
        d.append(item)
        insort(s, item)
        result.append(s[len(d) // 2])
    m = window_size // 2
    for item in seq:
        old = d.popleft()
        d.append(item)
        del s[bisect_left(s, old)]
        insort(s, item)
        result.append(s[m])
    return result


# current versions of code --------------------------------------


def get_cutout_bounds(obj, sector, window_size, cadence):

    # get the start and end times of a sector using the AWS data cube for cam 1 ccd 1
    with fits.open(
        f"s3://stpubdata/tess/public/mast/tess-s{str(sector).zfill(4)}-1-1-cube.fits",
        use_fsspec=True,
        fsspec_kwargs={"anon": True},
    ) as hdul:
        sector_start = hdul[0].header["TSTART"]
        sector_stop = hdul[0].header["TSTOP"]

    if cadence == '30m':
        stepsize='30m'
    elif cadence == '10m':
        stepsize='10m'
    elif cadence == '200s':
        # note that the 200s must be converted to days here
        stepsize = str(int(((sector_stop - sector_start)*86400)/200))


    print("querying ephemeris")
    horizons_query = Horizons(
        id=obj,
        location="@-95",  # the code for tess
        epochs={
            "start": Time(sector_start, format="btjd").iso,
            "stop": Time(sector_stop, format="btjd").iso,
            "step": stepsize,
        },
    )
    ephem = horizons_query.ephemerides(
        extra_precision=True, quantities="1"
    )  # just ra/dec

    print("checking TESS visibility")
    (
        outID,
        outEclipLong,
        outEclipLat,
        outSec,
        outCam,
        outCcd,
        outColPix,
        outRowPix,
        scinfo,
    ) = tess_stars2px_function_entry(
        starIDs=np.arange(len(ephem)).astype(int),
        starRas=ephem["RA"].data.filled(fill_value=np.nan),
        starDecs=ephem["DEC"].data.filled(fill_value=np.nan),
        trySector=sector,
    )
    if (len(outID) == 1):
        raise ValueError(f"Target '{obj}' not visible in sector {sector}")

    print("computing cutout bounds")
    lowest_c1 = (
        np.maximum((np.round(outColPix) - int(window_size // 2)), 0).min().astype(int)
    )
    highest_c2 = (
        np.minimum((np.round(outColPix) + int(window_size // 2)), 2048 + 44)
        .max()
        .astype(int)
    )
    lowest_r1 = (
        np.maximum((np.round(outRowPix) - int(window_size // 2)), 0).min().astype(int)
    )
    highest_r2 = (
        np.minimum((np.round(outRowPix) + int(window_size // 2)), 2048)
        .max()
        .astype(int)
    )

    cutout_size = (highest_c2 - lowest_c1, highest_r2 - lowest_r1)
    bounds = (lowest_c1, highest_c2, lowest_r1, highest_r2)

    # get ra, dec at a midpoint (when it's visible in the frame, so that obj_coord is visible)
    visible_ephem = ephem[outID]
    ra, dec = (
        visible_ephem[int(len(visible_ephem) / 2)]["RA"],
        visible_ephem[int(len(visible_ephem) / 2)]["DEC"],
    )
    obj_coord = SkyCoord(ra, dec, frame="icrs", unit="deg")
    return cutout_size, obj_coord, bounds


def get_cutout_Tesscut(obj_coord, sector, cam, ccd, size, root_dir):

    os.chdir(root_dir)
    cube_file = f"s3://stpubdata/tess/public/mast/tess-s{str(sector).zfill(4)}-{cam}-{ccd}-cube.fits"
    f = CutoutFactory().cube_cut(
        cube_file, coordinates=obj_coord, cutout_size=size, verbose=True, threads="auto"
    )

    with fits.open(f) as hdul:
        data_stack = hdul[1].data["FLUX"]
        times = hdul[1].data["TIME"]
        dquality = hdul[1].data["QUALITY"]

    times = Time(times, format="btjd")
    times_jd = times.jd

    return data_stack, times_jd, dquality


def download_cutout_frame(
    root_dir, sector, cam, ccd, target_name, cadence, buffer_pixel_size=30
):

    if target_name == "Uranus":
        obj_id = 799
    elif target_name == "Neptune":
        obj_id = 899
    else:
        obj_id = target_name

    cutout_size, obj_coord, bounds = get_cutout_bounds(
        obj_id, sector, buffer_pixel_size, cadence
    )

    data_stack, times_jd, dquality = get_cutout_Tesscut(
        obj_coord, sector, cam, ccd, cutout_size, root_dir
    )

    # translate size of data_stack to match rest of code
    data_stack = data_stack.transpose(1, 2, 0)

    return data_stack, times_jd, dquality, bounds


def save_raw_data_stack(
    root_dir, sector, cam, ccd, size_cutout, data_stack, times_jd, dquality, bounds
):  # , ephem):

    hdr = fits.Header()

    hdr["HDU1"] = "Raw image stack"
    hdr["HDU2"] = "Times (JD)"
    hdr["HDU3"] = "Data quality flags"
    hdr["HDU4"] = "Bounds of image cutout"
    # hdr['HDU5'] = "Ephemeris of object tracked in image"

    empty_primary = fits.PrimaryHDU(header=hdr)
    image_raw_hdu = fits.ImageHDU(data_stack)
    times_jd_hdu = fits.ImageHDU(times_jd)
    data_quality_hdu = fits.ImageHDU(dquality)
    bounds_hdu = fits.ImageHDU(bounds)
    # ephem_hdu = fits.ImageHDU(ephem)

    fits_origin_x, fits_origin_y = bounds[0], bounds[2]

    hdu = fits.HDUList(
        [empty_primary, image_raw_hdu, times_jd_hdu, data_quality_hdu, bounds_hdu]
    )  # , ephem_hdu])

    hdu.writeto(
        root_dir
        + "rawdatastack_s%i_cam%i_ccd%i_origin_%i_%i_size%i.fits"
        % (sector, cam, ccd, fits_origin_x, fits_origin_y, size_cutout),
        overwrite=True,
    )
    hdu.close()

    return hdu


def subtract_background(data_stack, percent_pix_scatteredlight=2):

    # get a scattered light profile to subtract from all pixels

    # find median flux of each pixel
    median_flux_time = np.median(
        data_stack, axis=2
    )  # (N_pix x N_pix grid of median fluxes)

    # take just the pixels with medians in the the lowest 5th percentile
    flux_max = np.percentile(median_flux_time, percent_pix_scatteredlight)
    ind1_lowmedianflux, ind2_lowmedianflux = np.where(median_flux_time < flux_max)
    pixel_profiles_Nth_percentile = data_stack[
        ind1_lowmedianflux, ind2_lowmedianflux, :
    ]

    # take the median profile of the pixels in the lowest 5th percentile
    scattered_light_profile = np.median(pixel_profiles_Nth_percentile, axis=0)

    # tile this and median profile
    tiled_scattered_light = np.tile(
        scattered_light_profile, (len(data_stack), len(data_stack[0]), 1)
    )
    tiled_median_flux = np.repeat(
        median_flux_time[:, :, np.newaxis], len(data_stack[0, 0]), axis=2
    )
    median_scattered_light_profile = tiled_scattered_light + (
        tiled_median_flux - np.median(tiled_scattered_light)
    )

    # subtract off this background profile from the data stack
    data_stack_tiled_sub = data_stack - median_scattered_light_profile

    return data_stack_tiled_sub, scattered_light_profile


def download_positions_NASAJPL(times_jd, sec, cam, ccd, target_name):

    # 799 is Uranus ID
    # 899 is Neptune ID

    if target_name == "Uranus":
        obj_id = 799
    elif target_name == "Neptune":
        obj_id = 899

    # query from tess location
    obj = Horizons(id=obj_id, location="500@-95", epochs=times_jd, id_type='name')
    eph = obj.ephemerides()
    ra, dec = eph["RA"], eph["DEC"]

    ticid = range(0, len(ra))
    (
        outID,
        outEclipLong,
        outEclipLat,
        outSec,
        outCam,
        outCcd,
        outColPix,
        outRowPix,
        scinfo,
    ) = tess_stars2px_function_entry(ticid, ra, dec)

    # vet to just the sec, cam, ccd that we want
    outColPix = outColPix[outSec == sec]
    outRowPix = outRowPix[outSec == sec]
    outCam = outCam[outSec == sec]
    outCcd = outCcd[outSec == sec]

    outColPix = outColPix[outCam == cam]
    outRowPix = outRowPix[outCam == cam]
    outCcd = outCcd[outCam == cam]

    outColPix = outColPix[outCcd == ccd]
    outRowPix = outRowPix[outCcd == ccd]

    positions = np.stack((outColPix, outRowPix))

    return positions


def query_positions_NASAJPL_batches(
    times_jd, sec, cam, ccd, target_name, batch_size=50
):

    # query NASA JPL in batches to avoid overflow errors

    # create empty array to hold positions
    positions_all = np.array([])

    # initialize indexing
    start_time_queried = 0
    end_time_queried = batch_size

    while start_time_queried < len(times_jd):

        positions = download_positions_NASAJPL(
            times_jd[start_time_queried:end_time_queried],
            sec=sec,
            cam=cam,
            ccd=ccd,
            target_name=target_name,
        )

        if len(positions_all) == 0:
            positions_all = positions
        else:
            positions_all = np.append(positions_all, positions, axis=1)

        start_time_queried += batch_size
        end_time_queried += batch_size

    return positions_all


def extract_positions(data_stack, times_jd, position_last_x, position_last_y):

    positions_all = np.array([0, 0])
    times_all = np.array([])
    frames_mask = np.ones(len(data_stack[0, 0]))

    #if exp_dur == 30:
    #    threshold = 10000
    #elif exp_dur == 10:
    #    threshold = 5000
    #elif exp_dur == 200:
    #    threshold = 800

    for data_frame_num in range(0, len(data_stack[0, 0])):
        data_frame = data_stack[:, :, data_frame_num]
        mean, median, std = sigma_clipped_stats(data_frame, sigma=3.0)
        daofind = DAOStarFinder(fwhm=2, threshold=5000)

        sources = daofind(data_frame - median)

        try:
            dist_last = np.sqrt(
                ((sources["xcentroid"] - position_last_x) ** 2.0)
                + (sources["ycentroid"] - position_last_y) ** 2.0
            )
            # xdist_last = abs(sources['xcentroid'].value - position_last_x)

            where_source = np.argmin(dist_last)

            if dist_last[where_source] < 20:

                if abs(sources["xcentroid"].value - position_last_x) < 20:
                    positions = np.array(
                        [
                            sources["xcentroid"][where_source],
                            sources["ycentroid"][where_source],
                        ]
                    )
                    positions_all = np.vstack((positions_all, positions))
                    times_all = np.append(times_all, times_jd[data_frame_num])

                    position_last_x, position_last_y = positions[0], positions[1]

                else:
                    frames_mask[data_frame_num] = 0

            else:
                frames_mask[data_frame_num] = 0

        except:
            frames_mask[data_frame_num] = 0

    positions_all = positions_all[1:]

    return positions_all, times_all, frames_mask


def filter_stack_to_found_positions(data_stack, frames_mask):

    inds_clean = np.where(frames_mask == 1)[0]
    data_stack_clean = data_stack[:, :, inds_clean]

    return data_stack_clean


def create_masked_stack(data_stack, positions_all, r_aperture):

    summed_fluxes_all = np.array([])

    pix_inds = np.asarray(list(range(0, len(data_stack))))
    pix_matrix = np.meshgrid(np.asarray(list(range(0, len(data_stack[0])))), pix_inds)

    data_stack_filtered = np.copy(data_stack)

    for frame_num in range(0, len(data_stack[0, 0])):

        position_x_frame, position_y_frame = (
            positions_all[frame_num][0],
            positions_all[frame_num][1],
        )
        data_frame = data_stack[:, :, frame_num]

        distances = np.sqrt(
            ((position_x_frame - pix_matrix[0]) ** 2.0)
            + ((position_y_frame - pix_matrix[1]) ** 2.0)
        )

        pixels_aperture = np.where((distances <= r_aperture))
        data_stack_filtered[pixels_aperture[0], pixels_aperture[1], frame_num] = 0

    masked_data_stack_filtered = np.ma.masked_equal(data_stack_filtered, 0)
    # median_image = np.ma.median(masked_data_stack_filtered, axis=2)

    return masked_data_stack_filtered


def remove_scattered_light_pixbypix(
    data_stack, r_aperture, outer_aperture, percent_pix_scatteredlight=10
):

    # make a copy of data stack to remove scattered light from
    scattered_light_stack = np.copy(data_stack)

    # indexing to get positions of all pixels
    pix_inds = np.asarray(list(range(0, len(data_stack))))
    pix_matrix = np.meshgrid(np.asarray(list(range(0, len(data_stack[0])))), pix_inds)

    print("Number of rows processed for scattered light subtraction...")

    for position_x_frame in range(0, len(data_stack)):

        if position_x_frame % 10 == 0:
            print("%i/%i" % (position_x_frame, len(data_stack)))
        # print(position_x_frame)

        for position_y_frame in range(0, len(data_stack[0])):

            # filter to only pixels moderately nearby
            distances = np.sqrt(
                ((position_x_frame - pix_matrix[1]) ** 2.0)
                + ((position_y_frame - pix_matrix[0]) ** 2.0)
            )
            pixels_nearby = np.where(
                (distances > r_aperture) & (distances < outer_aperture)
            )
            pixel_profiles_aperture = data_stack[pixels_nearby[0], pixels_nearby[1], :]

            # make masked stack and find median based on percentile fluxes
            flux_max_arr = np.percentile(
                pixel_profiles_aperture, percent_pix_scatteredlight, axis=0
            )
            pixel_profiles_aperture[pixel_profiles_aperture < flux_max_arr] = 0
            masked_stack_percentile = np.ma.masked_equal(pixel_profiles_aperture, 0)
            median_arr = bn_median(masked_stack_percentile, axis=0).data
            median_arr[np.isnan(median_arr)] = 0
            scattered_light_stack[position_x_frame, position_y_frame, :] = median_arr

    data_stack_reduced = data_stack - scattered_light_stack

    return data_stack_reduced, scattered_light_stack


def get_aperture_source_size(
    subtracted_image_stack, positions_vetted, frame_num=0, N_aperture_mult=10
):

    # use a reference frame (default: first image) to get the FWHM
    # for a source. Set the aperture to N_aperture_mult times that value.

    def gaussian(x, amplitude, mean, stddev):
        return amplitude * np.exp(-(((x - mean) / 4 / stddev) ** 2))

    image_flux_cut = subtracted_image_stack[
        :, int(positions_vetted[0, frame_num]), frame_num
    ]

    # get guess amplitude and mean based on where the peak is in the image
    amplitude_guess, mean_guess = np.max(image_flux_cut), np.argmax(image_flux_cut)
    popt, _ = optimize.curve_fit(
        gaussian,
        range(0, len(image_flux_cut)),
        image_flux_cut,
        p0=(amplitude_guess, mean_guess, 2),
    )

    # get the fwhm from the standard deviation
    stddev = popt[2]
    fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * stddev

    # set the aperture size to Nx the FWHM
    aperture_source = int(N_aperture_mult * fwhm)

    return aperture_source


def bn_median(masked_array, axis=None):
    """
    Perform fast median on masked array

    Parameters

    masked_array : `numpy.ma.masked_array`
        Array of which to find the median.

    axis : int, optional
        Axis along which to perform the median. Default is to find the median of
        the flattened array.
    """
    import numpy as np
    import bottleneck as bn

    data = masked_array.filled(fill_value=np.nan)
    med = bn.nanmedian(data, axis=axis)
    # construct a masked array result, setting the mask from any NaN entries
    return np.ma.array(med, mask=np.isnan(med))


def rolling_window_size_from_object_motion(
    aperture_source, positions_vetted, N_buffer_factor=4
):

    # get rate of motion (number of pixels traveled per day)
    size_source = aperture_source * 2
    pix_rate_per_cadence = np.sqrt(
        ((np.diff(positions_vetted[:, 0])) ** 2.0)
        + ((np.diff(positions_vetted[:, 1])) ** 2.0)
    )
    median_pix_rate_per_cadence = np.median(pix_rate_per_cadence)

    window_size = (
        int(N_buffer_factor * (aperture_source / median_pix_rate_per_cadence)) + 1
    )

    return window_size


def rolling_median_subtraction(
    data_stack,
    times_jd,
    positions_all,
    r_aperture=10,
    num_frames_median=700,
    window_size=50,
):

    masked_stack = create_masked_stack(data_stack, positions_all, r_aperture)

    # window smoothing within
    subtracted_image_stack = np.copy(data_stack)
    num_frames = len(data_stack[0, 0])

    for frame_ind in range(0, num_frames):

        if frame_ind % 300 == 0:
            print(frame_ind)

        inds_nearby_times = np.argsort(abs(times_jd - times_jd[frame_ind]))[
            :num_frames_median
        ]

        frames_at_nearby_times = masked_stack[:, :, inds_nearby_times]

        subtracted_image_stack[:, :, frame_ind] = data_stack[
            :, :, frame_ind
        ] - bn_median(frames_at_nearby_times, axis=2)

    return subtracted_image_stack


def save_stacks(
    sector,
    cam,
    ccd,
    root_dir,
    target_name,
    data_stack,
    times_jd,
    data_stack_vetted,
    subtracted_image_stack,
    positions_vetted,
    times_vetted,
    lc,
):

    hdr = fits.Header()

    hdr["HDU1"] = "Raw image stack"
    hdr["HDU2"] = "Times (JD)"
    hdr["HDU3"] = "Scattered light subtracted stack"
    hdr["HDU4"] = "Scattered light + median subtracted image stack"
    hdr["HDU5"] = "Target positions at vetted times (pixel, pixel)"
    hdr["HDU6"] = "Vetted times (JD)"
    hdr["HDU7"] = "Summed fluxes from extracted light curve"

    empty_primary = fits.PrimaryHDU(header=hdr)
    image_raw_hdu = fits.ImageHDU(data_stack)
    times_jd_hdu = fits.ImageHDU(times_jd)
    image_scattered_light_hdu = fits.ImageHDU(data_stack_vetted)
    image_reduced_hdu = fits.ImageHDU(subtracted_image_stack)
    positions_hdu = fits.ImageHDU(positions_vetted)
    times_vetted_hdu = fits.ImageHDU(times_vetted)
    lc_hdu = fits.ImageHDU(lc)

    hdu = fits.HDUList(
        [
            empty_primary,
            image_raw_hdu,
            times_jd_hdu,
            image_scattered_light_hdu,
            image_reduced_hdu,
            positions_hdu,
            times_vetted_hdu,
            lc_hdu,
        ]
    )

    hdu.writeto(
        root_dir + "stacks_%s_s%i_cam%i_ccd%i.fits" % (target_name, sector, cam, ccd),
        overwrite=True,
    )
    hdu.close()

    return hdu


def run_analysis_pipeline_solarsystem(
    root_dir,
    sector,
    cam,
    ccd,
    target_name,
    buffer_pixel_size,
    percent_pix_scatteredlight,
    diagnostic_images=True,
    animation=False,
):
    print('Processing... %s; sector %i, camera %i, CCD %i; BPS %i, PPS %i'
           %(target_name, sector, cam, ccd, buffer_pixel_size, percent_pix_scatteredlight))
    
    # make root directory if it doesn't exist yet
    if os.path.exists(root_dir) != True:
        os.makedirs(root_dir)

    if sector < 27:
        cadence = "30m"
    elif sector < 56:
        cadence = "10m"
    else:
        cadence = "200s"

    data_stack, times_jd, dquality, bounds = download_cutout_frame(
        root_dir, sector, cam, ccd, target_name, cadence, buffer_pixel_size
    )

    # get rid of nonzero data quality flag frames
    data_stack = data_stack[:, :, dquality == 0]
    times_jd = times_jd[dquality == 0]

    # get origin of fits file from bounds
    fits_origin_x, fits_origin_y = bounds[0], bounds[2]

    # make diagnostic image if requested
    if diagnostic_images == True:
        show_stack_frame(data_stack, root_dir, filename_ext="raw")

    # remove scattered light
    data_stack_reduced, scattered_light_stack = remove_scattered_light_pixbypix(
        data_stack,
        r_aperture=10,
        outer_aperture=20,
        percent_pix_scatteredlight=percent_pix_scatteredlight,
    )

    # make diagnostic image if requested
    if diagnostic_images == True:
        show_stack_frame(data_stack_reduced, root_dir, filename_ext="sl_sub")

    # query first position from JPL
    positions = query_positions_NASAJPL_batches(
        times_jd[:1], sec=sector, cam=cam, ccd=ccd, target_name=target_name
    )
    positions[0, :] -= fits_origin_x
    positions[1, :] -= fits_origin_y
    position_last_x, position_last_y = positions[0, 0], positions[1, 0]

    # get empirical planet positions; remove frames where not found ---------------

    # simple background subtraction to get locations
    data_stack_sub, scattered_light_profile = subtract_background(data_stack)

    # find planet positions
    positions_vetted, times_vetted, frames_mask = extract_positions(
        data_stack_sub, times_jd, position_last_x, position_last_y
    )

    # filter to only positions where planet was found
    data_stack_vetted = filter_stack_to_found_positions(data_stack_reduced, frames_mask)

    # -----------------------------------------------------------------------------

    # get aperture size for rolling window
    aperture_source = get_aperture_source_size(data_stack_vetted, positions_vetted)
    print("Aperture size used for source: %i pixels" % (aperture_source))

    # determine window size for rolling median subtraction
    num_frames_median = rolling_window_size_from_object_motion(
        aperture_source, positions_vetted, N_buffer_factor=4
    )

    # carry out median subtraction
    subtracted_image_stack = rolling_median_subtraction(
        data_stack_vetted,
        times_vetted,
        positions_vetted,
        num_frames_median=num_frames_median,
    )

    # get light curve
    lc = get_summed_fluxes(
        subtracted_image_stack, positions_vetted, r_aperture=aperture_source
    )


    # save it all into a stack
    save_stacks(sector, cam, ccd, root_dir, target_name, data_stack, times_jd, data_stack_vetted, subtracted_image_stack, positions_vetted, times_vetted, lc)

    # save light curve into numpy txt file
    np.savetxt(root_dir+'lc_%s_s%i_cam%i_ccd%i.txt' %(target_name, sector, cam, ccd), np.vstack((times_vetted, lc)))


    # make diagnostic image + light curve if requested
    if diagnostic_images == True:
        try:
            show_stack_frame(subtracted_image_stack, root_dir, filename_ext="sl_med_sub")
        except:
            print('Failed to produce reduced image visual. Moving on...')

        try:
            plot_lightcurve(target_name, root_dir, times_vetted, lc)
        except:
            print('Failed to produce light curve diagnostic. Moving on...')

        try:
            plot_shift_lightcurve_diagnostic(
                target_name,
                root_dir,
                subtracted_image_stack,
                times_vetted,
                positions_vetted,
                aperture_source,
                full_scale=False,
            )
        except:
            print('Failed to produce shifted light curve diagnostic; \
                   field of view may be too small. Moving on...')

        # plot Lomb-Scargle periodogram
        try:
            frequency, power, false_alarm = find_frequencies_ls(times_vetted, lc,\
                                            target_name, sector, cam, ccd, root_dir, \
                                            extra_data=True)
        except:
            print('Failed to produce Lomb-Scargle periodogram. Moving on...')

        # plot Welch's power spectrum
        try:
            plot_welchs_power_spectrum(times_vetted, lc, root_dir)
        except:
            print("Failed to produce Welch's power spectrum. Moving on...")

        try:
            star_proximity_tracker(
                target_name, root_dir, data_stack_vetted, positions_vetted, times_vetted, lc
            )
        except:
            print('Failed to produce star proximity tracker diagnostic. \
                   Lack of bright nearby stars in field? Moving on...')

        try:
            comparison_flux_nearby_star(
                target_name,
                root_dir,
                data_stack_vetted,
                times_vetted,
                lc,
                aperture_source,
                threshold=5000,
            )
        except:
            print('Failed to produce comparison flux for nearby bright star; \
                   there may be none above threshold in view.')


    if animation == True:
        # save animation
        try:
            make_animation_2panel(
                data_stack,
                subtracted_image_stack,
                times_vetted,
                root_dir,
                vmin1=0,
                vmax1=1000,
                vmin2=0,
                vmax2=1000,
            )
        except:
            print('Failed to make 2-panel animation. Moving on...')

        # make Lomb-Scargle moving window animation
        # by default, set window to 1/3 the total number
        # of points in the light curve
        try:
            window_size_ls = int(0.33*len(lc))

            make_animation_lombscargle(
                times_vetted,
                lc,
                window_size_ls,
                root_dir=root_dir,
            )
        except:
            print('Failed to make Lomb-Scargle animation. Moving on...')

    print('Reduction completed, %s; sector %i, camera %i, CCD %i; BPS %i, PPS %i'
           %(target_name, sector, cam, ccd, buffer_pixel_size, percent_pix_scatteredlight))

    return None


# CHANGE ONLY THESE LINES -----------------

# setup
target_name = 'C/2025 N1'
sector = 92
cam = 1
ccd = 2
buffer_pixel_size = 40
percent_pix_scatteredlight = 10

# set root directory for results
root_dir = '/scratch11/ktp9/TESS_SolarSystem/%s/s%i_cam%i_ccd%i/results_bps%i_pps%i/' %(target_name, sector, cam, ccd, buffer_pixel_size, percent_pix_scatteredlight)

run_analysis_pipeline_solarsystem(
    root_dir,
    sector,
    cam,
    ccd,
    target_name,
    buffer_pixel_size,
    percent_pix_scatteredlight,
    diagnostic_images=True,
    animation=False,
)

# # ----------------------------------------
