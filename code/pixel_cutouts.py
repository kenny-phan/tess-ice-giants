import os
from cutout import makeCutout

fillst = '/scratch11/ktp9/DIA/70/weirdffis70.txt'
ffidir = '/scratch11/ktp9/DIA/70/ffis70/'
cutdir = '/scratch11/ktp9/DIA/70/cutouts70/'

position = (1979.058, 1092.890)
cutout_size = (150, 150)
hdu_num = 1

with open(fillst, 'r') as filelist:
    filenames = filelist.read().splitlines()

for filename in filenames:
    file_path = os.path.join(ffidir, filename)
    ctpath = cutdir+'ct-'+filename
    makeCutout(file_path, position, cutout_size, ctpath, hdu_num)
