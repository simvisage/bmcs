'''
Created on 13.06.2019

@author: hspartali
'''

from scipy.signal import savgol_filter

import numpy as np


def prepend_zero(array):
    return np.concatenate((np.zeros((1)), array))


def smooth_ascending_disp_branch(disp_ascending, disp_rest, force_extrema_indices):

    disp_ascending = savgol_filter(disp_ascending, window_length=51, polyorder=2)
    
    disp_rest = disp_rest[force_extrema_indices]
    disp = np.concatenate((disp_ascending, disp_rest)) 
    return disp


def skip_extra_displacement():
    temp = 0

# testing
# force = np.array(pd.read_csv('C:\\Users\\hspartali\\Desktop\\CT80-35.csv', delimiter=';', decimal=',', skiprows=4, usecols=[1]))
# disp = np.array(pd.read_csv('C:\\Users\\hspartali\\Desktop\\CT80-35.csv', delimiter=';', decimal=',', skiprows=4, usecols=[3]))
# skip_noise_of_ascending_branch(force, disp)

# ar = np.array((5, 6, 7, 8, 9))
# 
# print(prepend_zero(ar))
