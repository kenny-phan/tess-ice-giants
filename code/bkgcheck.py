#pull an arbitrary pixel from your cutouts, plot to confirm systematics have been factored out

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

#for file parsing
import time
import glob, os
from os import listdir
from os.path import isfile, join

cutdir = '/scratch11/ktp9/DIA/70/cutouts70/'
clndir = '/scratch11/ktp9/DIA/70/clean70'
cdedir = '/home/ktp9/TESSNeptune24/code/'

def arbPixel(fits_file, x, y, hdu_num):

    # Open the FITS file
    with fits.open(fits_file) as hdul:
        # Print the structure of the FITS file
        hdul.info()

        # Assuming the image data is in the primary HDU (HDU 0)
        image_data = hdul[hdu_num].data

        # Access the flux at a specific pixel (e.g., pixel at row 100, column 200)
        flux = image_data[x, y]
    
    return flux

def plotFlux(): 

