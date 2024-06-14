#creating cutouts from full ffis

#for cutout creation
from astropy.nddata import Cutout2D
from astropy.io import fits
from astropy.wcs import WCS

#for file parsing
import time
import glob, os
from os import listdir
from os.path import isfile, join

# directories with full ffis & directory to put cutouts
ffidir = '/scratch11/ktp9/DIA/ffis70/'
cutdir = '/scratch11/ktp9/DIA/cutouts70/'
cdedir = '/home/ktp9/TESSNeptune24/code/'

def makeCutout(fits_file, ra, dec, cutout_size, output_file, hdu_num):
    # Load the image and WCS information
    with fits.open(fits_file, mode='readonly') as hdulist:
        wcs_info = WCS(hdulist[hdu_num].header, relax=True)
        cal_image = hdulist[hdu_num].data
        header = hdulist[hdu_num].header
        
    # Convert the RA and Dec coordinates to pixel coordinates
    position = wcs_info.world_to_pixel_values(ra, dec)

    # Create the cutout
    cutout = Cutout2D(cal_image, position, cutout_size, wcs=wcs_info)

    # Update the header with the cutout WCS information
    cutout_header = cutout.wcs.to_header()
    new_header = header.copy()
    new_header.update(cutout_header)
        
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
ra = 356.2175
dec = -3.2784
cutout_size = (150, 150)
hdu_num = 1 #can be 0, 1, 2 -- I'm confused about this one but usually 0 or 1 work!

for ii in range(0, nfiles):
    hld = files[ii].split('.')
    finnme = hld[0]+'_ct.fits'

    if (os.path.isfile(cutdir+finnme)==0):
        st = time.time()
        sts = time.strftime("%c")
        print('Now cutting '+files[ii]+' at '+sts+'.')
        
        fits_file = ffidir + finnme
        output_file = cutdir + finnme
        makeCutout(fits_file, ra, dec, cutout_size, output_file, 1)
        fn = time.time()
        print('Cutout for '+files[ii]+' finished in '+str(fn-st)+'seconds.')

print('All done! After a while, crocodiel!')
