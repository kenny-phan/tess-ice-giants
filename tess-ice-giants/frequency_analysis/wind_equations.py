import numpy as np
from scipy.special import eval_legendre
from scipy.interpolate import interp1d

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
