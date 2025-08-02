import os
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
from scipy.ndimage import label, center_of_mass
from photutils.aperture import CircularAperture, aperture_photometry

# ~~~ UTILITY FUNCTIONS ~~~ #

def print_head_entry_fits(folder, header_string, header_hdu=1):
    """
    Prints the same header entry of every .fits file in a folder
    """
    
    fits_files = [f for f in os.listdir(folder) if f.endswith('.fits')]

    for fname in fits_files:
        fpath = os.path.join(folder, fname)
        try:
            with fits.open(fpath, memmap=True) as hdul:
                header = hdul[header_hdu].header
                value = header.get(header_string, 'header string not found')
                print(f"{fname}: {value}")
        except Exception as e:
            print(f"{fname}: Error reading file - {e}")

def show_all_fits_images(folder, data_hdu=1, header_hdu=1, vmin=None, vmax_percentile=None):
    """
    Loops through all .fits files in the folder and displays each image using imshow.
    
    Parameters:
        folder (str): Path to folder containing .fits files.
    """
    fits_files = [f for f in os.listdir(folder) if f.lower().endswith('.fits')]

    for fname in fits_files:
        file_path = os.path.join(folder, fname)

        with fits.open(file_path) as hdul:
            data = hdul[data_hdu].data
            header = hdul[header_hdu].header
            
            print(f"/.n--- {fname} ---")
            
            plt.figure()
            if vmin is not None:
                plt.imshow(data, origin='lower', cmap='gray', vmin=vmin, vmax=np.percentile(data, vmax_percentile))
            else:
                plt.imshow(data, origin='lower', cmap='gray')#, vmin=0, vmax=np.percentile(data, 99.99))
            plt.title(fname)
            plt.colorbar(label='Pixel value')
            plt.show()

def show_image_list(image_list, filter_name='Unknown', automatic_vrange=True):
    for i, img in enumerate(image_list):
        plt.figure()
        if automatic_vrange:
            plt.imshow(img, origin='lower', cmap='gray')
        else:
            plt.imshow(img, origin='lower', cmap='gray', vmin=0, vmax=np.percentile(img, 99))
        plt.title(f"Filter: {filter_name}, Image #{i+1}")
        plt.colorbar(label='Pixel value')
        plt.show()

# ~~ IMAGE PROCESSING ~~ #

# 1. take important data from .fits files and put into a dictionary for ease of use, subtract sky

def fits_to_list(folder, time_string, ra_string, dec_string, orientat_string, data_hdu=1, header_hdu=0, alt_header=None):
    """
    Loops through all .fits files in the folder, processes them, 
    and groups them into a list of dictionaries.

    Args:
        folder (str): Folder containing .fits files
        time_string (str): Header keyword for time (e.g. 'MJD-OBS')
        ra_string (str): Header keyword for RA (e.g. 'RA_TARG')
        dec_string (str): Header keyword for Dec (e.g. 'DEC_TARG')
        data_hdu (int): Index of the HDU where the image data lives
        header_hdu (int): Index of the HDU from which to read the header

    Returns:
        list of dict: Each dict contains 'data', 'mjd', 'ra', 'dec'
    """
    fits_files = [f for f in os.listdir(folder) if f.lower().endswith('.fits')]
    fits_list = []

    for fname in fits_files:
        file_path = os.path.join(folder, fname)

        try:
            with fits.open(file_path, memmap=True) as hdul:
                data = hdul[data_hdu].data
                header = hdul[header_hdu].header          

                mjd_obs = header.get(time_string, np.nan)
                ra = header.get(ra_string, np.nan)
                dec = header.get(dec_string, np.nan)
                orientat = header.get(orientat_string, np.nan)

                if alt_header is not None:
                    alt_head = hdul[alt_header].header
                    orientat = alt_head.get(orientat_string, np.nan)
                
                if data is not None:
                    # Normalize the data
                    sky_subtracted = data - np.nanmedian(data)

                    fits_list.append({
                        'data': sky_subtracted,
                        'mjd': mjd_obs,
                        'ra': ra,
                        'dec': dec,
                        'orientat': orientat
                    })
        except Exception as e:
            print(f"Error processing {fname}: {e}")

    return fits_list

# 2. get aperture, flux 

def draw_largest_bright_circle(images, use_adaptive_threshold=True,
                                percentile_threshold=95,
                                radius_scale=1.3, show=True):
    """
    Detects the largest contiguous bright region and draws a circular aperture around it.

    Parameters:
        images (list of np.ndarray): List of 2D image arrays.
        use_adaptive_threshold (bool): Use dynamic thresholding instead of fixed percentile.
        percentile_threshold (float): Percentile for fixed brightness threshold (if not adaptive).
        radius_scale (float): Factor to expand aperture radius estimate.
        show (bool): Whether to display the results.

    Returns:
        list of dicts with 'position', 'radius', 'flux'
    """
    results = []

    for i, image in enumerate(images):
        # Threshold the image
        if use_adaptive_threshold:
            median = np.nanmedian(image)
            upper_val = np.nanpercentile(image, 95)
            threshold = median + 0.5 * (upper_val - median)
        else:
            threshold = np.nanpercentile(image, percentile_threshold)

        mask = image > threshold

        # Label connected components
        labeled, num_features = label(mask)
        if num_features == 0:
            results.append({'position': None, 'radius': None, 'flux': None})
            continue

        # Find the largest contiguous region
        sizes = [(labeled == i + 1).sum() for i in range(num_features)]
        largest_idx = np.argmax(sizes)
        largest_region_mask = (labeled == largest_idx + 1)

        # Estimate position and radius
        yx_center = center_of_mass(largest_region_mask)
        base_radius = np.sqrt(sizes[largest_idx] / np.pi)
        radius_estimate = base_radius * radius_scale

        aperture = CircularAperture(yx_center[::-1], r=radius_estimate)
        flux = aperture_photometry(image, aperture, method='exact')['aperture_sum'][0]

        results.append({
            'position': yx_center[::-1],
            'radius': radius_estimate,
            'flux': flux
        })

        if show:
            plt.figure(figsize=(6, 5))
            plt.imshow(image, origin='lower', cmap='gray',
                       vmin=np.nanpercentile(image, 5), vmax=np.nanpercentile(image, 99))
            aperture.plot(color='red', lw=2)
            plt.title(f"Image {i}: Brightest Circle (Threshold = {threshold:.1f})")
            plt.axis('off')
            plt.show()

    return results

# ~~ CONVERT MJD TO BTJD ~~

from astropy.coordinates import SkyCoord, EarthLocation, GCRS
from astropy.time import Time
import astropy.units as u
from astropy.constants import c

def hst_light_travel_time_correction(ra_deg, dec_deg, mjd):
    """
    Compute the light travel time correction from a target to HST, in days.

    Parameters:
    -----------
    ra_deg : float
        Right Ascension of the target in degrees.
    dec_deg : float
        Declination of the target in degrees.
    mjd : float
        Modified Julian Date (UTC)

    Returns:
    --------
    delta_t : float
        Light travel time correction in days.
    """

    time = Time(mjd, format='mjd', scale='utc')

    # Target with arbitrary distance
    target = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, distance=1*u.pc, frame='icrs').transform_to(GCRS(obstime=time))

    # Approximate HST location in low Earth orbit
    hst_location = EarthLocation(lat=28.5383*u.deg, lon=-80.567*u.deg, height=540*u.km)
    hst_gcrs = hst_location.get_gcrs(time)
    hst_coord = SkyCoord(hst_gcrs)

    # 3D separation and light travel time
    distance = target.separation_3d(hst_coord)
    light_time = distance / (c * 86400)

    return light_time.to(u.day).value

def mjd_to_btjd(mjd, light_correction):
    return mjd + 2400000.5 + light_correction - 2457000.0

def convert_mjd_list(mjd_list, ra_list, dec_list, light_corr_func):
    btjd_list = []
    for ii, mjd in enumerate(mjd_list):
        light_correction = light_corr_func(ra_list[ii], dec_list[ii], mjd_list[ii])
        btjd_list.append(mjd_to_btjd(mjd_list[ii], light_correction))

    return btjd_list

