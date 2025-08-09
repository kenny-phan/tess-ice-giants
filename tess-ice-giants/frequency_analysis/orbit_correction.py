import numpy as np
from astroquery.jplhorizons import Horizons
from astropy.time import Time 

def get_lightcurve_info(flux_data_file):
    times, raw_flux = np.loadtxt(flux_data_file)

    start_date = Time(times[0], format='jd').iso
    end_date   = Time(times[-1], format='jd').iso
    step       = str(len(times) - 1) 

    return times, raw_flux, start_date, end_date, step

def get_delta(object_id, start_date, end_date, step, location='@sun'):
    obj = Horizons(id=object_id, location=location, epochs={'start': start_date, 'stop': end_date, 'step': step})
    eph = obj.ephemerides()
    return eph['delta']

def correct_lightcurve(raw_flux, delta):
    
    correction_array = (delta / np.mean(delta))**2

    return np.multiply(raw_flux, correction_array)