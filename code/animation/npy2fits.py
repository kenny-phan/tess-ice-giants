from astropy.io import fits
import numpy as np

def save_stacks(data_stack, data_stack_tiled_sub, times_jd, file_path):

    hdr = fits.Header()
    
    hdr['HDU1'] = "Raw image stack"
    hdr['HDU2'] = "Fully reduced image stack"
    hdr['HDU3'] = "Times (JD)"
    
    empty_primary = fits.PrimaryHDU(header=hdr)
    image_raw_hdu = fits.ImageHDU(data_stack)
    image_reduced_hdu = fits.ImageHDU(data_stack_tiled_sub)
    times_hdu = fits.ImageHDU(times_jd)


    hdu = fits.HDUList([empty_primary, image_raw_hdu, image_reduced_hdu, times_hdu])
    
    hdu.writeto(file_path, overwrite=True)
    hdu.close()
    
    return hdu

stkdir = '/scratch11/ktp9/DIA/70/stacks/'
data_stack = np.load(stkdir+'dirty.npy')
data_stack_tiled_sub = np.load(stkdir+'malenasubtract.npy')
time_jd = 
save_stacks(data_stack, data_stack_tiled_sub, )