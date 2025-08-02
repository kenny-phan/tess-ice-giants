import numpy as np
import matplotlib.pyplot as plt
from astroquery.jplhorizons import Horizons
from astropy.time import Time 

# --- User Parameters ---
target_id = '799'           # Can be 'Neptune', 'Jupiter', 'Ceres', etc.
observer_id = '@tess'          # Can be any spacecraft, like 'JWST', 'Spitzer', etc.
flux_data_file = '/scratch11/ktp9/TESS_SolarSystem/Uranus/s44_cam1_ccd4/results_bps40_pps10/lc_Uranus_s44_cam1_ccd4.txt'

reference_distance = 19.19 #29.07      # avg distance between target and earth

# --- Functions ---
def ra_dec_to_cartesian(ra, dec, r):
    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)
    x = r * np.cos(dec_rad) * np.cos(ra_rad)
    y = r * np.cos(dec_rad) * np.sin(ra_rad)
    z = r * np.sin(dec_rad)
    return np.array([x, y, z])

def corrected_flux(raw_flux, relative_distances, reference_distance):
    return raw_flux * ((relative_distances / reference_distance) ** 2)

def get_ephemeris(object_id, start_date, end_date, step, location='@sun'):
    obj = Horizons(id=object_id, location=location, epochs={'start': start_date, 'stop': end_date, 'step': step})
    eph = obj.ephemerides()
    return eph['RA'], eph['DEC'], eph['r']

# --- Load Flux Data ---
times, raw_flux = np.loadtxt(flux_data_file)

start_date = Time(times[0], format='jd').iso
end_date   = Time(times[-1], format='jd').iso
step       = str(len(times) - 1)            # In seconds or time string ('1d', '10m', etc.)
btjd_offset = 0           # Kepler time system specific

# --- Get Ephemeris Data ---
print('Getting observer ephemeris...')
obs_ra, obs_dec, obs_r = get_ephemeris(observer_id, start_date, end_date, step)

print('Getting target ephemeris...')
tgt_ra, tgt_dec, tgt_r = get_ephemeris(target_id, start_date, end_date, step)

# --- Compute Cartesian Positions ---
print('Converting RA/Dec to Cartesian...')
obs_xyz = ra_dec_to_cartesian(obs_ra, obs_dec, obs_r)
tgt_xyz = ra_dec_to_cartesian(tgt_ra, tgt_dec, tgt_r)

# --- Compute Relative Distance ---
print('Calculating relative distances...')
relative_distances = np.sqrt(np.sum((obs_xyz - tgt_xyz)**2, axis=0))

# --- Apply Flux Correction ---
print('Correcting flux...')
corrected_flux = corrected_flux(raw_flux, relative_distances, reference_distance)

# --- Save Result ---
output_file = f'/scratch11/ktp9/TESS_SolarSystem/{observer_id.lower()}_{target_id.lower()}_flux_corrected_s44.npz'
np.savez(output_file, times=times, flux=corrected_flux)

print(f"All done! Flux saved to {output_file}")
