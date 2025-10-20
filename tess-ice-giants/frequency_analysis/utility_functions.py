import numpy as np

def z_score(data):
    return (data - np.nanmedian(data)) / np.nanstd(data)

def chi2(O, E):
    return np.nansum((O - E)**2 / E)