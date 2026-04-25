import numpy as np
from scipy.special import eval_legendre
from scipy.interpolate import interp1d

def radius_phi(R_eq, R_p):
        
    def R_phi(phi):
        return R_eq / np.sqrt(1 + (np.tan(phi)*(R_p/R_eq))**2)
    
    return R_phi

# frequency to wind speed equation
def frequency_wind_speed(Req, Rp, P): # R radius in m, P period in hours
    fr = 24 / P 
    R_phi = radius_phi(Req, Rp)
    def equation(phi, f):
        return 2 * np.pi * R_phi(phi) * np.cos(phi) * (f - fr) / 86400
    return equation

# Neptune Equations

def six_order_fit(a=0, b=0, c=0, d=0):
    def model_eqn(phi):
        return a + b * np.degrees(phi)**2 + c * np.degrees(phi)**4 + d * np.degrees(phi)**6
    return model_eqn

# error for neptune eqns
def sigma_six_order_fit(sigma_a=0, sigma_b=0, sigma_c=0, sigma_d=0):

    def equation(phi):
        return np.sqrt(sigma_a**2 + (phi*sigma_b)**2 + (phi**2*sigma_c)**2 + (phi**2*sigma_d)**2)
    
    return equation

# Copy + paste the following to get equations from Sromovsky+ 1993 & Tollefson+ 2018
"""
sromovsky1993_four = six_order_fit(-398, 1.88e-1, -1.2e-5)
sromovsky1993_six = six_order_fit(-389, 1.53e-1, 1.01e-5, -3.1e-9)

tollefson2013_h = six_order_fit(-325, 1.58e-1, -1.21e-5)
tollefson2013_kp = six_order_fit(-415, 2.35e-1, -2.23e-5)
tollefson2014_h = six_order_fit(-292, 1.45e-1, -1.18e-5)
tollefson2014_kp = six_order_fit(-433, 2.4e-1, -2.73e-5)
"""

sromovsky1993_four = six_order_fit(-398, 1.88e-1, -1.2e-5)
sromovsky1993_four_err = sigma_six_order_fit(12, 1.4e-2, 3e-6)

sromovsky1993_six = six_order_fit(-389, 1.53e-1, 1.01e-5, -3.1e-9)
sromovsky1993_six_err = sigma_six_order_fit(13, 2.1e-2, 0.72e-5, 0.7e-9)

tollefson2013_kp = six_order_fit(-415, 2.35e-1, -2.23e-5)
tollefson2013_kp_err = sigma_six_order_fit(42, 5.34e-2, 1.14e-5)

tollefson2014_kp = six_order_fit(-433, 2.4e-1, -2.73e-5)
tollefson2014_kp_err = sigma_six_order_fit(56, 7.88e-2, 1.9e-5)

# Uranus Equations

# Karkoschka 1998
def karkoshka1998_N(phi):
    R = 25559000
    fr = 17.24
    eq = 482 - 8*np.sin(phi) + 127*((np.sin(phi))**2)
    eq_m_s = eq * (2*np.pi*R*np.cos(phi)) / (360*24*60*60) # convert deg/day to m/s
    return eq_m_s - (2*np.pi*R*np.cos(phi)) / (fr*60*60) # account for rotation

def karkoshka1998_S(phi):
    return karkoshka1998_N(-phi)

# Hammel+ 2001
def hammel2001(phi):
    return 170 * (0.6 * np.cos(phi) - np.cos(3 * phi)) # in m/s

# Sromovsky+ 2012c 

def Rphi(phi): # eqn 5 in S2012c
    Re = 25559
    Rp = 24973
    return Re / np.sqrt(1 + (np.tan(phi)*(Rp/Re))**2)

def sromovsky2012_even(phi): # for 1997-2011
    coeffs = [1.25037831, 0, 3.72050211, 0, 0.12041514, 0, 
              -0.73555624, 0, -0.34206461, 0, 0.20553095, 0, 0.08364616]
    
    sin_phi = np.sin(phi)
    leg_sum = sum(c * eval_legendre(l, sin_phi) for l, c in enumerate(coeffs)) # eqn 3
    
    return leg_sum * 4.8481e-3 * Rphi(phi) # eqn 4

def sromovsky2012_odd_N(phi): # for 1997-2011
    coeffs = [1.24197012, -0.02848715, 3.69457598, 0.08752786, 
              0.15287708, -0.13202142, -0.65646542, -0.08292523, 
              -0.31793598, 0.09172810, 0.15934681, 0.06504432, 0.02752308]
    
    sin_phi = np.sin(phi)
    leg_sum = sum(c * eval_legendre(l, sin_phi) for l, c in enumerate(coeffs)) # eqn 3
    
    return leg_sum * 4.8481e-3 * Rphi(phi) # eqn 4

def sromovsky2012_odd_S(phi):
    return sromovsky2012_odd_N(-phi)

## sromovsky2012 error propagation
def sigma_Rphi(Re = 25559, Rp = 24973, sigma_Re = 4, sigma_Rp = 20):

    def eqn(phi):
        dRphi_dRe = (1 / np.sqrt(1 + (np.tan(phi)*(Rp/Re))**2)) * (1 + (np.tan(phi)*(Rp/Re))**2)**(-3/2) * (np.tan(phi)*(Rp/Re))**2 / Re
        dRphi_dRp = (1 + (np.tan(phi)*(Rp/Re))**2)**(-3/2) * (np.tan(phi)**2 * Rp / Re)
        return np.sqrt((dRphi_dRe * sigma_Re)**2 + (dRphi_dRp * sigma_Rp)**2)

    return eqn

def sigma_sromovsky2012_odd_N(phi):
    coeffs = [1.24197012, -0.02848715, 3.69457598, 0.08752786, 
              0.15287708, -0.13202142, -0.65646542, -0.08292523, 
              -0.31793598, 0.09172810, 0.15934681, 0.06504432, 0.02752308]
    
    sin_phi = np.sin(phi)
    leg_sum = sum(c * eval_legendre(l, sin_phi) for l, c in enumerate(coeffs)) # eqn 3
    constant = 4.8481e-3

    dU_dRphi = constant * leg_sum
    dU_dleg_sum = constant * Rphi(phi)

    reperr = 0.088 # degrees/h, pg. 11 of Sromovsky+ 2012c
    dr = sigma_Rphi()

    return np.sqrt((dU_dRphi * dr(phi))**2 + (dU_dleg_sum * reperr)**2)

def sigma_sromovsky2012_odd_S(phi):
    return sigma_sromovsky2012_odd_N(-phi)

## Sromovsky+ 2015

def make_sromovsky2015():
    points = [0.00, 9.07, 18.15, 27.21, 36.27, 45.31, 54.34, 63.35, 72.33, 81.29,
              90.23, 99.13, 108.00, 116.83, 125.62, 134.36, 143.06, 151.72, 160.32, 168.86,
              177.35, 185.77, 194.14, 202.43, 210.53, 218.84, 226.82, 234.33, 241.20, 247.23,
              252.26, 256.13, 258.70, 259.87, 259.57, 257.78, 254.54, 249.90, 243.99, 236.94,
              228.93, 220.13, 210.74, 200.94, 190.88, 180.72, 170.58, 160.54, 150.67, 140.99,
              131.53, 122.27, 113.20, 104.30, 95.56, 86.97, 78.54, 70.28, 62.22, 54.39,
              46.83, 39.60, 32.73, 26.26, 20.20, 14.58, 9.40, 4.63, 0.26, -3.75,
              -7.43, -10.82, -13.96, -16.88, -19.60, -22.16, -24.55, -26.78, -28.85, -30.75,
              -32.49, -34.05, -35.43, -36.63, -37.64, -38.49, -39.16, -39.67, -40.02, -40.22,
              -40.24, -40.11, -39.85, -39.42, -38.81, -37.99, -36.95, -35.66, -34.12, -32.29,
              -30.17, -27.74, -24.99, -21.92, -18.53, -14.81, -10.77, -6.41, -1.76, 3.19,
              8.41, 13.87, 19.57, 25.47, 31.55, 37.76, 44.08, 50.48, 56.92, 63.39,
              69.87, 76.35, 82.83, 89.33, 95.88, 102.50, 109.26, 116.20, 123.37, 130.83,
              138.62, 146.78, 155.32, 164.22, 173.44, 182.92, 192.53, 201.83, 210.02, 217.83,
              225.25, 232.26, 238.84, 239.62, 240.08, 240.23, 240.04, 239.52, 238.66, 232.26,
              225.77, 219.19, 212.53, 205.78, 198.95, 192.04, 185.05, 177.98, 170.85, 169.86,
              168.32, 166.19, 163.48, 160.17, 160.37, 159.58, 156.88, 153.20, 148.50, 142.75,
              137.54, 130.97, 122.96, 113.47, 102.43, 89.78, 75.45, 56.61, 37.75, 18.88,
              0.000]

    res = points[::-1]
    eqn = interp1d(np.arange(-90, 91, 1), res, kind='cubic')

    def f(phi):
        return eqn(np.degrees(phi))

    return f

sromovsky2015_N = make_sromovsky2015()

def sromovsky2015_S(phi):
    return sromovsky2015_N(-phi)

def make_sromovsky2015_degrees():
    points = [4.100, 4.100, 4.100, 4.100, 4.100, 4.100, 4.100, 4.100, 4.100, 4.100,
              4.100, 4.100, 4.100, 4.100, 4.100, 4.100, 4.100, 4.100, 4.100, 4.100,
              4.100, 4.100, 4.100, 4.100, 4.097, 4.100, 4.099, 4.090, 4.072, 4.043,
              4.002, 3.946, 3.875, 3.789, 3.687, 3.571, 3.443, 3.302, 3.153, 2.997,
              2.836, 2.673, 2.510, 2.349, 2.191, 2.039, 1.893, 1.753, 1.620, 1.493,
              1.373, 1.258, 1.149, 1.045, 0.946, 0.850, 0.759, 0.672, 0.588, 0.509,
              0.434, 0.363, 0.298, 0.237, 0.181, 0.129, 0.083, 0.040, 0.002, -0.032,
              -0.064, -0.092, -0.118, -0.142, -0.164, -0.185, -0.204, -0.222, -0.238, -0.253,
              -0.266, -0.278, -0.289, -0.298, -0.305, -0.312, -0.317, -0.321, -0.323, -0.325,
              -0.325, -0.324, -0.322, -0.319, -0.314, -0.308, -0.300, -0.290, -0.278, -0.264,
              -0.247, -0.228, -0.206, -0.181, -0.154, -0.124, -0.090, -0.054, -0.015, 0.027,
              0.072, 0.120, 0.170, 0.223, 0.278, 0.335, 0.394, 0.455, 0.518, 0.582,
              0.647, 0.714, 0.783, 0.854, 0.927, 1.002, 1.081, 1.164, 1.253, 1.346,
              1.447, 1.554, 1.669, 1.793, 1.924, 2.064, 2.210, 2.359, 2.501, 2.645,
              2.790, 2.937, 3.086, 3.166, 3.247, 3.328, 3.410, 3.492, 3.575, 3.578,
              3.581, 3.585, 3.588, 3.591, 3.595, 3.598, 3.601, 3.605, 3.608, 3.749,
              3.891, 4.035, 4.181, 4.328, 4.596, 4.869, 5.120, 5.377, 5.638, 5.904,
              6.250, 6.605, 6.970, 7.344, 7.729, 8.124, 8.530, 8.530, 8.530, 8.530,
              8.530
              ]
    
    res = points[::-1]
    eqn = interp1d(np.arange(-90, 91, 1), res, kind='cubic')

    def f(phi):
        return eqn(np.degrees(phi))
    
    return f

sromovsky2015_degrees_N = make_sromovsky2015_degrees()

def sromovsky2015_degrees_S(phi):
    return sromovsky2015_degrees_N(-phi)

# error for sromovsky2015
def sigma_sromovsky2015_N(phi):
    constant = 4.8481e-3
    dphi_dt = sromovsky2015_degrees_N(phi)

    dU_dRphi = constant * dphi_dt
    dU_ddphi_dt = constant * Rphi(phi)

    dr = sigma_Rphi()
    ddphi_dt = 0.147 # degrees/day, pg. 11 of Sromovsky+ 2015
    # note: the above "reperr" only relates to the Legendre fit used 
    # latitudes 46S to 67N, and does not neccesarily reflect the entire 
    # error of that portion. Nonetheless, we use it as an estimate.

    return np.sqrt((dU_dRphi * dr(phi))**2 + (dU_ddphi_dt * ddphi_dt)**2)

def sigma_sromovsky2015_S(phi):
    return sigma_sromovsky2015_N(-phi)

## MCMC SAMPLING EQUATIONS ##

## these need to be an equation for emcee sampling
def PHI(model_eqn, Re, Rp, P): # R radius in m, P period in hours
    fr = 24/P 
    R_phi = radius_phi(Re, Rp)

    def equation(phi, f):
        return model_eqn(phi) / (R_phi(phi) * (f - fr))

    return equation

## RHS of equation 1 after rearrangement
def RHS():
    def equation(phi):
        return 2*np.pi*np.cos(phi) / 86400
    
    return equation

# these also have to be equations.. i think.. where model_eqn and model_eqn_err are functions of phi
# vc stands for variance contribution
def vc_u(R, P, model_eqn_err): # wind eqn vc
    fr = 24/P

    def equation(phi, f):
        return (model_eqn_err(phi) / (R(phi) * (f - fr)))**2

    return equation

def vc_R(model_eqn, R, P, R_err):
    fr = 24/P

    def equation(phi, f):
        return (R_err(phi) * model_eqn(phi) / (R(phi)*R(phi)*(f - fr)))**2
    
    return equation

def vc_f(model_eqn, R, P): # f_err is NOT constant! one f_err for each f!
    fr = 24/P

    def equation(phi, f, f_err):
        return (f_err * model_eqn(phi) / (R(phi) * (f - fr)**2))**2
    
    return equation

def vc_fr(model_eqn, R, P, P_err): # f_err is NOT constant! one f_err for each f!
    fr = 24/P
    fr_err = P_err / P**2 #error propagate for P -> fr

    def equation(phi, f):
        return (fr_err * model_eqn(phi) / (R(phi) * (f - fr)**2))**2
    
    return equation

def sigma(model_eqn, Re, Rp, P, model_eqn_err, Re_err, Rp_err, P_err): # outputs sigma as a function of phi, f, and f_err
    
    R = radius_phi(Re, Rp)
    R_err = sigma_Rphi(Re=Re, Rp=Rp, sigma_Re=Re_err, sigma_Rp=Rp_err) # R_err is a function of phi

    def equation(phi, f, f_err):
        vc_uu = vc_u(R, P, model_eqn_err)
        vc_RR = vc_R(model_eqn, R, P, R_err)
        vc_ff = vc_f(model_eqn, R, P)
        vc_frfr = vc_fr(model_eqn, R, P, P_err)

        return np.sqrt(vc_uu(phi, f) + vc_RR(phi, f) + vc_ff(phi, f, f_err) + vc_frfr(phi, f))
        
    return equation
