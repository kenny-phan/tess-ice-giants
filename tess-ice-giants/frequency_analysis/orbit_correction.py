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

def run_orbit_correction(target_id, observer_id, flux_data_file):
    print("Parsing Light Curve")
    times, raw_flux, start_date, end_date, step = get_lightcurve_info(flux_data_file)

    print("Getting ephemeris")
    delta = get_delta(target_id, start_date, end_date, step, observer_id)

    print("Correcting Light curve")
    corrected_lightcurve = correct_lightcurve(raw_flux, delta)

    return times, raw_flux, corrected_lightcurve   