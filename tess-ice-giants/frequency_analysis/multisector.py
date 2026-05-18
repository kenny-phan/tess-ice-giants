import numpy as np
import matplotlib.pyplot as plt

from astropy.timeseries import LombScargle

# def z_score(data):
#     return (data - np.nanmedian(data)) / np.nanstd(data)

# def chi2(O, E):
#     return np.nansum((O - E)**2 / E)

def get_models(time, flux, peak_freqs):
        ls = LombScargle(time, flux)
        models = []
        for i in range(len(peak_freqs)):
                best_frequency = peak_freqs[i]
                # t0 = time[0]  # reference epoch n42['times'], n42['corrections']
                model = ls.model(time, best_frequency)
                models.append(model)

        return models

def find_percent_variability(flux, models):
        flux_mean = np.nanmean(flux)
        flux_std = np.nanstd(flux)
        flux_variability = 100 * flux_std / flux_mean 

        model_variabilities = []
        flux_minus_model_variabilities = []

        for model in models:  
                model_mean = np.nanmean(model)
                model_max = np.nanmax(model)

                model_variability = 100 * (model_max - model_mean) / model_mean
                model_variabilities.append(model_variability)

                flux_minus_model = flux - model
                flux_minus_model_std = np.nanstd(flux_minus_model)
                flux_minus_model_variability = 100 * flux_minus_model_std / flux_mean
                flux_minus_model_variabilities.append(flux_minus_model_variability)

        return flux_variability, model_variabilities, flux_minus_model_variabilities


