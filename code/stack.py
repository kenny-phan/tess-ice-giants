#stack up those fits files!

#fits analysis
import numpy as np
from astropy.io import fits

#file parsing
from tqdm import tqdm
import glob, os
from os import listdir
from os.path import isfile, join

fitdir = '/scratch11/ktp9/DIA/70/cutouts/' 
cdedir = '/home/ktp9/TESSNeptune24/code/'
stkdir = '/scratch11/ktp9/DIA/70/stacks/' #clndir + 'stack.npy'

#axis size in px
x = 150
y = 150

#get the image list and the number of files which need cutoutsn
os.chdir(fitdir) #changes to the raw image direcotory
files = [f for f in glob.glob("*.fits") if isfile(join(fitdir, f))] #gets the relevant files with the proper extension
nfiles = len(files)
files.sort()
os.chdir(cdedir) #changes back to the code directory

btchsze = 50
btchno = 0
imgs = []
times = []

for i in tqdm(range(0, nfiles, btchsze)):
    btchfils = files[i:i+btchsze]
    for file in btchfils:
        f = fitdir + file
        with fits.open(f) as opnfits:
            flx = opnfits[0].data
            imgs.append(flx)
            start = opnfits[0].header['TSTART']
            stop = opnfits[0].header['TSTOP']
            time = (start + stop)/2
            times.append(time)
            del flx, start, stop
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
np.save(stkdir+'time.npy', times)

for file in stacks:
    os.remove(file)

print(f'Here is your stack with shape = {fullstk.shape}, at times with length {len(times)}.')
print('All done! Hit the road, happy taod!')



