#for cutout creation
from astropy.nddata import Cutout2D
from astropy.io import fits
from astropy.wcs import WCS

#for file parsing
import time
import glob, os
from os import listdir
from os.path import isfile, join

from tqdm import tqdm

# directories with full ffis & directory to put cutouts
ffidir = '/scratch11/ktp9/DIA/42/ffis42/'
cutdir = '/scratch11/ktp9/DIA/42/cutouts/'
cdedir = '/home/ktp9/TESSNeptune24/code/'

def makeCutout(fits_file, position, cutout_size, output_file, hdu_num):
    # Load the image and WCS information
    with fits.open(fits_file, mode='readonly') as hdulist:
        #wcs_info = WCS(hdulist[hdu_num].header, relax=True)
        cal_image = hdulist[hdu_num].data
        header = hdulist[hdu_num].header
        
    # Convert the RA and Dec coordinates to pixel coordinates
    #position = wcs_info.world_to_pixel_values(ra, dec)

    # Create the cutout
    cutout = Cutout2D(cal_image, position, cutout_size) #, wcs=wcs_info

    # Update the header with the cutout WCS information
    #cutout_header = cutout.wcs.to_header()
    new_header = header.copy()
    #new_header.update(cutout_header)
        
    # Create a PrimaryHDU object with the cutout data and the updated header
    hdu = fits.PrimaryHDU(data=cutout.data, header=new_header)
    hdulist_new = fits.HDUList([hdu])

    # Write the cutout to a new FITS file
    hdulist_new.writeto(output_file, overwrite=True)

#get the image list and the number of files which need cutoutsn
os.chdir(ffidir) #changes to the raw image direcotory
files = [f for f in glob.glob("*.fits") if isfile(join(ffidir, f))] #gets the relevant files with the proper extension
files.sort()
nfiles = len(files)
os.chdir(cdedir) #changes back to the code directory

#variables, change as needed
#ra, dec deprecated in favor of pixel vals as found in TESS Playground
#ra = 356.567526
#dec = -3.278378
position = (309.907, 264.214)#sector 70: (1979.058, 1092.890)
cutout_size = (256, 256)
hdu_num = 1 #can be 0, 1, depending on where the fits is from

for ii in tqdm(range(0, nfiles)):
    hld = files[ii].split('.')
    finnme = hld[0]+'_ct.fits'

    if (os.path.isfile(cutdir+finnme)==0):
        st = time.time()
        sts = time.strftime("%c")
        print('Now cutting '+files[ii]+' at '+sts+'.')
        
        fits_file = ffidir+hld[0]+'.fits'
        output_file = cutdir+finnme
        makeCutout(fits_file, position, cutout_size, output_file, 1)
        fn = time.time()
        print('Cutout for '+files[ii]+' finished in '+str(fn-st)+'seconds.')

print('All done! See you later, alliagator!')

