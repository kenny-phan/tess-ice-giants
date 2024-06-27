#check for the noise clean removes

import numpy as np

clnfil = '/scratch11/ktp9/DIA/70/stacks/full.npy' 
#mstdir =
rawfil = '/scratch11/ktp9/DIA/70/stacks/dirty.npy'
optfil = '/scratch11/ktp9/DIA/70/stacks/removed.npy' 

def subtract(path1, path2, outputfile=None):
    data1 = np.load(path1)
    data2 = np.load(path2)

    result = data1 - data2

    if outputfile:
        np.save(outputfile, result)

    return result

subtract(rawfil, clnfil, optfil)


