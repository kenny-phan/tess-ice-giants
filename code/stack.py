#stack up those fits files!

#fits analysis
import numpy as np
from astropy.io import fits
from astropy.time import Time

#file parsing
from tqdm import tqdm
import glob, os
from os import listdir
from os.path import isfile, join

fitdir = '/scratch11/ktp9/DIA/42/cutouts/' 
cdedir = '/home/ktp9/TESSNeptune24/code/'
stkdir = '/scratch11/ktp9/DIA/42/stacks/' #clndir + 'stack.npy'

#UPDATE: axis size in px (same as positions)
x = 256
y = 256

#get the image list and the number of files which need cutoutsn
os.chdir(fitdir) #changes to the raw image direcotory
files = [f for f in glob.glob("*.fits") if isfile(join(fitdir, f))] #gets the relevant files with the proper extension
nfiles = len(files)
files.sort()
os.chdir(cdedir) #changes back to the code directory

btchsze = 50
btchno = 0
imgs = []
btjd_times = []
utc_times = []

for i in tqdm(range(0, nfiles, btchsze)):
    btchfils = files[i:i+btchsze]
    for file in btchfils:
        f = fitdir + file
        with fits.open(f) as opnfits:
            #append flux
            flx = opnfits[0].data
            imgs.append(flx)
            
            #append BTJD time
            btjd_start = opnfits[0].header['TSTART']
            btjd_stop = opnfits[0].header['TSTOP']
            average_time_btjd = (btjd_start + btjd_stop)/2
            btjd_times.append(average_time_btjd)
            
            #append utc time
            utc_start = opnfits[0].header['DATE-OBS']
            utc_stop = opnfits[0].header['DATE-END']
            utc = [utc_start, utc_stop]
            utc_objects = Time(utc, format='isot', scale='utc')
            julian_dates = utc_objects.jd
            average_jd = np.mean(julian_dates)
            average_time_utc = Time(average_jd, format='jd', scale='utc')
            utc_times.append(average_time_utc.isot)
            
            #save some memory
            del btjd_start, btjd_stop, utc_start, utc_stop
    
    btjd_times_array = np.array(btjd_times, dtype=float)
    utc_times_array = np.array(utc_times, dtype=str)
    
    times = np.zeros((2, len(btjd_times)), dtype=object)
    times[0, :] = btjd_times_array
    times[1, :] = utc_times_array
    
    #combine times into one array
    #times = np.vstack([btjd_times_array, utc_times_array])    
    
    stk = np.stack(imgs, axis=2)
    imgs = []
    np.save(stkdir+'count%03d.npy'%(btchno), stk)
    btchno += 1
    
os.chdir(stkdir)
stacks = [file for file in os.listdir(stkdir) if 'count' in file]
stacks.sort()

holder = []

for file in tqdm(stacks):
    path = os.path.join(stkdir, file)
    load = np.load(path)
    holder.append(load)

fullstk = np.concatenate(holder, axis=2)
np.save(stkdir+'raw.npy', fullstk)
np.save(stkdir+'time.npy', times, allow_pickle=True)

for file in stacks:
    os.remove(file)

print(f'Here is your stack with shape = {fullstk.shape}, at times with length {len(times)}.')
print('All done! In a while, corcodile!')



