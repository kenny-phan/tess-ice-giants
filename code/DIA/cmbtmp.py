#this program will combine images to make a master frame

#if you use this code, please cite Oelkers et al. 2015, AJ, 149, 50

#import the relevant libraries for basic tools
from astropy.io import fits
import numpy as np
import scipy
from scipy import stats
from os import path
import math
import time

#import relevant libraries for a list
import glob, os
from os import listdir
from os.path import isfile, join, exists

###UPDATE HERE#####
#what field are you looking at?
sector = '70'
camera = '1'
ccd = '1'

#useful directories
cdedir = '/home/ktp9/TESSNeptune24/code/' #code directory
mstdir = '/scratch11/ktp9/DIA/70/master/henchmen/' #directory where the cleaned images reside
findir = '/scratch11/ktp9/DIA/70/master/voltron/' #directory for the final master frame
###END UPDATE###

#get the image list and the number of files which need reduction
os.chdir(mstdir) #changes to the image direcotory
files = [f for f in glob.glob(camera+"_"+ccd+"_*.fits") if isfile(join(mstdir, f))] #gets the relevant files with the proper extension
files.sort()
nfiles = len(files)
os.chdir(cdedir) #changes back to the code directory

#set up the holder for the final fiel count
nx = fits.getval(mstdir+files[0], 'NAXIS2')
ny = fits.getval(mstdir+files[0], 'NAXIS1')
all_data = np.ndarray(shape=(nfiles,nx,ny))
expt = np.zeros(nfiles)
num = np.zeros(nfiles)

for ii in range(0,nfiles):

	#read in the image
	img_data = fits.getdata(mstdir+files[ii])
	expt[ii] = fits.getval(mstdir+files[ii],'EXPTIME')
	num[ii] = fits.getval(mstdir+files[ii],'NUMCOMB')

	#add the image to the vector
	all_data[ii] = img_data 

	if (ii % 10 == 0) and (ii > 0):
		print('Finished with 10 images at '+str(time.strftime("%a %d %b %Y %H:%M:%S"))+'.')

#median combine the data
combined_data = np.median(all_data,axis=0)

# Write data to new file    
new_image = fits.PrimaryHDU(combined_data)
new_image.header.set('NUMCOMB', np.sum(num))
new_image.header.set('EXPTIME', np.median(expt))

#print the file with the appropriate counter
new_image.writeto(findir+sector+'_'+camera+'_'+ccd+'_master_py.fits',overwrite=True)

print("The master frame was created using a median of "+str(np.sum(num))+" images.")

del all_data, img_data # clear up some memory
print("All done. Hasta manana, igauna!")
