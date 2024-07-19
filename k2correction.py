import numpy as np
import matplotlib.pyplot as plt

#ephemeris
from astroquery.jplhorizons import Horizons

#define functions
def ra_dec_to_cartesian(ra, dec, r):
    # Convert degrees to radians
    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)
    
    x = r * np.cos(dec_rad) * np.cos(ra_rad)
    y = r * np.cos(dec_rad) * np.sin(ra_rad)
    z = r * np.sin(dec_rad)
    
    return np.array([x, y, z])

def k2_corrected_flux(raw_flux, relative_distances, reference_distance):
    corrected_flux = raw_flux * ((relative_distances / reference_distance) ** 2)
    return corrected_flux

with open('/scratch11/ktp9/DIA/phot.m2.20160224.cjp.dat', 'r') as file:
    lines = file.readlines()

# Process the lines as needed
k2 = []
for line in lines:
    # Assume space-separated values, adjust as needed
    values = line.strip().split()
    k2.append([float(v) for v in values])

k2_array = np.array(k2)
timesk2 = k2_array[:,0] - 7.5 #(BTJD offset)
fluxk2 = k2_array[:,1]

#get the ephemeris
start_date = '2014-12-01'
end_date = '2015-01-19'
step = '68480'

print('getting k2 ephemeris')
# K2 spacecraft ephemeris
k2_id = 'Kepler (spacecraft)'
k2_obj = Horizons(id=k2_id, location='@sun', epochs={'start': start_date, 'stop': end_date, 'step': step})
k2_eph = k2_obj.ephemerides()

print('getting neptune ephemeris')
# Neptune ephemeris
neptune_id = '899'
neptune_obj = Horizons(id=neptune_id, location='@sun', epochs={'start': start_date, 'stop': end_date, 'step': step})
neptune_eph = neptune_obj.ephemerides()

print('ectracting ra dec')
# Extract RA and Dec
k2_ra = k2_eph['RA']
k2_dec = k2_eph['DEC']
neptune_ra = neptune_eph['RA']
neptune_dec = neptune_eph['DEC']

k2_r = 1.03
neptune_r = 30.07
reference_distance = 29.07

print('comvertinf to cartesian')
# Convert to Cartesian coordinates
k2_cartesian = ra_dec_to_cartesian(k2_ra, k2_dec, k2_r)
neptune_cartesian = ra_dec_to_cartesian(neptune_ra, neptune_dec, neptune_r)

print('calculatin\'')
# Calculate relative distances
relative_distances = np.sqrt(np.sum((k2_cartesian - neptune_cartesian)**2, axis=0))

corrected_fluxk2 = k2_corrected_flux(fluxk2, relative_distances, reference_distance)
np.save('/scratch11/ktp9/DIA/k2_flux_distcorrected', corrected_fluxk2)
print("All done! Bye-bye butterfly!")
