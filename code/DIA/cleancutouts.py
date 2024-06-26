# modified for cutouts, clean.py from DIA, thanks to Oelkers et al. 2015, AJ, 149, 50
#it will also align the images to the first image in the list

#import the relevant libraries for basic tools
import numpy as np
import scipy
from scipy import stats
import scipy.ndimage as ndimage
import astropy
from astropy.io import fits
from astropy.nddata.utils import Cutout2D
from astropy.wcs import WCS
import math
import time

#libraries for image registration
import FITS_tools
from FITS_tools.hcongrid import hcongrid

#import relevant libraries for a list
import glob, os
from os import listdir
from os.path import isfile, join
from tqdm import tqdm 

#import relevant spline libraries
from scipy.interpolate import Rbf

#####UPDATE INFORMATION HERE####
align = 1 # yes = 1 no = 0 to align based on coordinates

#useful directories
rawdir = '/scratch11/ktp9/DIA/70/ffis70/' #'/scratch11/ktp9/DIA/cutouts/' #directory with the raw images
cdedir = '/home/ktp9/TESSNeptune24/code/' #directory where the code 'lives'
clndir = '/scratch11/ktp9/DIA/70/cleanffis70/'#directory for the cleaned images to be output

#sample every how many pixels? usually 32x32 is OK but it can be larger or smaller
pix = 32 # UPDATE HERE FOR BACKGROUND SPACING
axs = 2048 # UPDATE HERE FOR IMAGE AXIS SIZE
###END UPDATE INFORMATION###

#get the image list and the number of files which need reduction
os.chdir(rawdir) #changes to the raw image direcotory
files = [f for f in glob.glob("*.fits") if isfile(join(rawdir, f))] #gets the relevant files with the proper extension
files.sort()
nfiles = len(files)
os.chdir(cdedir) #changes back to the code directory

#get the zeroth image for registration
#read in the image
ref, rhead = fits.getdata(rawdir+files[0], header = True)
rhead['CRPIX1'] = axs/2 #central x-pixel
#cutout dimensions
rhead['NAXIS1'] = axs 
rhead['NAXIS2'] = axs

#sample every how many pixels?
bxs = 512 #how big do you want to make the boxes for each image?, ie each "box" is a square section, cleaned individually
lop = 2*pix
sze = int((bxs/pix)*(bxs/pix)+2*(bxs/pix)+1) #size holder for later
#begin cleaning
for ii in tqdm(range(0, nfiles)):
    hld = files[ii].split('.')
    finnme = hld[0]+'_sa.fits'
    #only create the files that don't exist
    if (os.path.isfile(clndir+finnme) == 0):
        #start the watch
        st = time.time()
        sts = time.strftime("%c")
        print('Now cleaning '+files[ii]+' at '+sts+'.')

        #read in the image
        orgimg, header = fits.getdata(rawdir+files[ii], header = True)  
        w = WCS(header)
        cut = Cutout2D(orgimg, (axs/2, axs/2), (axs, axs), wcs = w) #centerpoint in parantheses
        bigimg = cut.data
    
        #update the header
        header['CRPIX1'] = axs/2 #same deal as above
        header['NAXIS1'] = axs
        header['NAXIS2'] = axs
    
        #get the holders ready
        res = np.zeros(shape=(axs, axs)) #holder for the background 'image'
        bck = np.zeros(shape=(int((axs/bxs)**2))) #get the holder for the image backgroudn
        sbk = np.zeros(shape=(int((axs/bxs)**2))) #get the holder for the sigma of the image background
        tts = 0
        for oo in range(0, axs, bxs):
            for ee in range(0, axs, bxs):
                img = bigimg[ee:ee+bxs, oo:oo+bxs] #split the image into small subsections
        
                #calculate the sky statistics
                cimg, clow, chigh = scipy.stats.sigmaclip(img, low=2.5, high = 2.5) #do a 2.5 sigma clipping
                sky = np.median(cimg) #determine the sky value
                sig = np.std(cimg) #determine the sigma(sky)
        
                bck[tts] = sky #insert the image median background
                sbk[tts] = sig #insert the image sigma background
        
                #create holder arrays for good and bad pixels
                x = np.zeros(shape=(sze))
                y = np.zeros(shape=(sze))
                v = np.zeros(shape=(sze))
                s = np.zeros(shape=(sze))
                nd = int(0)
	
                #begin the sampling of the "local" sky value
                for jj in range(0, bxs+pix, pix):
                    for kk in range(0,bxs+pix, pix):
                        il = np.amax([jj-lop,0])
                        ih = np.amin([jj+lop, bxs-1])
                        jl = np.amax([kk-lop, 0])
                        jh = np.amin([kk+lop, bxs-1])
                        c = img[jl:jh, il:ih]
                        #select the median value with clipping
                        cc, cclow, cchigh = scipy.stats.sigmaclip(c, low=2.5, high = 2.5) #sigma clip the background
                        lsky = np.median(cc) #the sky background
                        ssky = np.std(cc) #sigma of the sky background
                        x[nd] = np.amin([jj, bxs-1]) #determine the pixel to input
                        y[nd] = np.amin([kk, bxs-1]) #determine the pixel to input
                        v[nd] = lsky #median sky
                        s[nd] = ssky #sigma sky
                        nd = nd + 1

                #now we want to remove any possible values which have bad sky values
                rj = np.where(v <= 0) #stuff to remove
                kp = np.where(v > 0) #stuff to keep

                if (len(rj[0]) > 0):
                	#keep only the good points
                    xgood = x[kp]
                    ygood = y[kp]
                    vgood = v[kp]
                    sgood = s[kp]

                    print(rj)
                    print(len(rj[0]))
                    for jj in range(0, len(rj[0])):
                        #select the bad point
                        xbad = x[rj[0][jj]]
                        ybad = y[rj[0][jj]]
                        #use the distance formula to get the closest points
                        rd = np.sqrt((xgood-ygood)**2.+(ygood-ybad)**2.)
                        #sort the radii
                        pp = sorted(range(len(rd)), key = lambda k:rd[k])
                        #use the closest 10 points to get a median
                        vnear = vgood[pp[0:9]]
                        ave = np.median(vnear)
                        #insert the good value into the array
                        v[rj[0][jj]] = ave

                #now we want to remove any possible values which have bad sigmas
                rjs = np.where(s >= 2*sig)
                rj  = rjs[0]
                kps = np.where(s < 2*sig)
                kp  = kps[0]
                
                if (len(rj) > 0):
                    #keep only the good points
                    xgood = np.array(x[kp])
                    ygood = np.array(y[kp])
                    vgood = np.array(v[kp])
                    sgood = np.array(s[kp])
                    
                    for jj in range(0, len(rj)):
                        #select the bad point
                        xbad = x[rj[jj]]
                        ybad = y[rj[jj]]
                        #use the distance formula to get the closest points
                        rd = np.sqrt((xgood-xbad)**2.+(ygood-ybad)**2.)
                        #sort the radii
                        pp = sorted(range(len(rd)), key = lambda k:rd[k])
                        #use the closest 10 points to get a median
                        vnear = vgood[pp[0:9]]
                        ave = np.median(vnear)
                        #insert the good value into the array
                        v[rj[jj]] = ave

                #now we interpolate to the rest of the image with a thin-plate spline	
                xi = np.linspace(0, bxs-1, bxs)
                yi = np.linspace(0, bxs-1, bxs)
                XI, YI = np.meshgrid(xi, yi)
                rbf = Rbf(x, y, v, function = 'thin-plate', smooth = 0.0)
                reshld = rbf(XI, YI)
    	
                #now add the values to the residual image
                res[ee:ee+bxs, oo:oo+bxs] = reshld
                tts = tts+1

        #get the median background
        mbck = np.median(bck)
        sbck = np.median(sbk)
	
        #subtract the sky gradient and add back the median background
        sub = bigimg-res
        sub = sub + mbck
        
        # Define the required WCS keys
        required_wcs_keys = ['CTYPE1', 'CTYPE2', 'CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2', 'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2']
        
        # Function to ensure all WCS keys are present
        def ensure_wcs_keys(header, reference_header):
            for key in required_wcs_keys:
                if key not in header:
                    if key in reference_header:
                        header[key] = reference_header[key]
                    else:
                        header[key] = 'UNKNOWN'
                if key not in reference_header:
                    if key in header:
                        reference_header[key] = header[key]
                    else:
                        reference_header[key] = 'UNKNOWN'
        
        # Ensure both headers have all necessary WCS information
        ensure_wcs_keys(header, rhead)
    
        # Align the image only if WCS information is present and valid
        if all(key in header for key in required_wcs_keys) and all(key in rhead for key in required_wcs_keys):
            algn = hcongrid(sub, header, rhead)
        else:
            print(f"Skipping alignment for {files[ii]} due to missing or invalid WCS information.")
            algn = sub
        
        # Update the header with WCS information from the reference header
        for key in required_wcs_keys:
            header[key] = rhead[key]
        
        # Write out the subtraction
        shd = fits.PrimaryHDU(algn, header=header)
        shd.writeto(clndir + finnme, overwrite=True)
    	
        #stop the watch
        fn = time.time()
        print('Background subtraction for '+files[ii]+' finished in '+str(fn-st)+'s.')

print('All done! See ya later alliagtor.')
    
