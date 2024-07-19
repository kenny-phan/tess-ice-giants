from astropy.io import fits
import numpy as np

def save_stacks(data_stack_tiled_sub, times, file_path): #data_stack, 

    hdr = fits.Header()
    
    #hdr['HDU1'] = "Raw image stack"
    hdr['HDU1'] = "Fully reduced image stack"
    hdr['HDU2'] = "Times BTJD"
    
    empty_primary = fits.PrimaryHDU(header=hdr)
    #image_raw_hdu = fits.ImageHDU(data_stack)
    image_reduced_hdu = fits.ImageHDU(data_stack_tiled_sub)
    times_hdu = fits.ImageHDU(times)


    hdu = fits.HDUList([empty_primary, image_reduced_hdu, times_hdu]) #image_raw_hdu,
    
    hdu.writeto(file_path, overwrite=True)
    hdu.close()
    
    return hdu

stkdir = '/scratch11/ktp9/DIA/70/archive/'
svefil = '/scratch11/ktp9/DIA/70/archive/DIA70.fits'
#data_stack = np.load(stkdir+'raw.npy')
data_stack_tiled_sub = np.load(stkdir+'raw_DIA.npy')
times = np.load(stkdir+'time_DIA.npy')
save_stacks(data_stack_tiled_sub, times, svefil) #data_stack, 

print(f"Raw, cleaned, and time data saved to {svefil}. Be sweet, parakete!")
