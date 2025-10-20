import numpy as np
from astroquery.jplhorizons import Horizons
from astropy.time import Time 

def get_lightcurve_info(flux_data_file):
    times, raw_flux = np.loadtxt(flux_data_file)

    start_date = Time(times[0], format='jd').iso
    end_date   = Time(times[-1], format='jd').iso
    step       = str(len(times) - 1) 

    return times, raw_flux, start_date, end_date, step

def get_delta(target_id, start_date, end_date, step, observer_id):
    obj = Horizons(id=target_id, location=observer_id, epochs={'start': start_date, 'stop': end_date, 'step': step})
    eph = obj.ephemerides()
    return eph['delta']

def correct_lightcurve(raw_flux, delta):
    
    correction_array = (delta / np.mean(delta))**2

    return np.multiply(raw_flux, correction_array)

def run_orbit_correction(target_id, observer_id, flux_data_file, crop_range=None):
    print("Parsing Light Curve")
    times, raw_flux, start_date, end_date, step = get_lightcurve_info(flux_data_file)

    if crop_range:
        mask = (times >= times[crop_range[0]]) & (times <= times[crop_range[1]])
        times = times[~mask]
        raw_flux = raw_flux[~mask]

        start_date = Time(times[0], format='jd').iso
        end_date   = Time(times[-1], format='jd').iso
        step       = str(len(times) - 1) 
        
    print("Getting ephemeris")
    delta = get_delta(target_id, start_date, end_date, step, observer_id)

    print("Correcting Light curve")
    corrected_lightcurve = correct_lightcurve(raw_flux, delta)

    return times, raw_flux, corrected_lightcurve   


#duplicate funtions for K2 data, as I am lazy 
def get_lightcurve_info_k2(flux_data_file):

    # julian date of Dec 1st, 2014
    jd_dec1_2014 = 2456992.5

    k2 = np.loadtxt(flux_data_file)
    times, raw_flux = k2[:,0] + jd_dec1_2014, k2[:,1]

    start_date = Time(times[0], format='jd').iso
    end_date   = Time(times[-1], format='jd').iso
    step       = str(len(times) - 1) 

    return times, raw_flux, start_date, end_date, step

def run_orbit_correction_k2(target_id, observer_id, flux_data_file, crop_range=None):
    if observer_id is None:
        observer_id = '500@-143'  # K2 Earth-trailing orbit
    print("Parsing Light Curve")
    times, raw_flux, start_date, end_date, step = get_lightcurve_info_k2(flux_data_file)

    if crop_range:
        mask = (times >= times[crop_range[0]]) & (times <= times[crop_range[1]])
        times = times[~mask]
        raw_flux = raw_flux[~mask]

        start_date = Time(times[0], format='jd').iso
        end_date   = Time(times[-1], format='jd').iso
        step       = str(len(times) - 1) 
        
    print("Getting ephemeris")
    delta = get_delta(target_id, start_date, end_date, step, observer_id)

    print("Correcting Light curve")
    corrected_lightcurve = correct_lightcurve(raw_flux, delta)

    return times, raw_flux, corrected_lightcurve   