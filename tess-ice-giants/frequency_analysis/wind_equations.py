import numpy as np
from scipy.special import eval_legendre

# frequency to wind speed equation
def frequency_wind_speed(R, P): # R radius in m, P period in hours
    fr = 24 / P 
    def equation(phi, f):
        return 2 * np.pi * R * np.cos(phi) * (f - fr) / 86400
    return equation

# Neptune Equations

def six_order_fit(a=0, b=0, c=0, d=0):
    def model_eqn(phi):
        return a + b * np.degrees(phi)**2 + c * np.degrees(phi)**4 + d * np.degrees(phi)**6
    return model_eqn

# Copy + paste the following to get equations from Sromovsky+ 1993 & Tollefson+ 2018
"""
sromovsky1993_four = six_order_fit(-398, 1.88e-1, -1.2e-5)
sromovsky1993_six = six_order_fit(-389, 1.53e-1, 1.01e-5, -3.1e-9)

tollefson2013_h = six_order_fit(-325, 1.58e-1, -1.21e-5)
tollefson2013_kp = six_order_fit(-415, 2.35e-1, -2.23e-5)
tollefson2014_h = six_order_fit(-292, 1.45e-1, -1.18e-5)
tollefson2014_kp = six_order_fit(-433, 2.4e-1, -2.73e-5)
"""

# Uranus Equations

# Karkoschka 1998
def karkoshka1998(phi):
    R = 25559000
    fr = 17.24
    eq = 482 - 8*np.sin(phi) + 127*((np.sin(phi))**2)
    eq_m_s = eq * (2*np.pi*R*np.cos(phi)) / (360*24*60*60) # convert deg/day to m/s
    return eq_m_s - (2*np.pi*R*np.cos(phi)) / (fr*60*60) # account for rotation

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

def sromovsky2012_odd(phi): # for 1997-2011
    coeffs = [1.24197012, -0.02848715, 3.69457598, 0.08752786, 
              0.15287708, -0.13202142, -0.65646542, -0.08292523, 
              -0.31793598, 0.09172810, 0.15934681, 0.06504432, 0.02752308]
    
    sin_phi = np.sin(phi)
    leg_sum = sum(c * eval_legendre(l, sin_phi) for l, c in enumerate(coeffs)) # eqn 3
    
    return leg_sum * 4.8481e-3 * Rphi(phi) # eqn 4

