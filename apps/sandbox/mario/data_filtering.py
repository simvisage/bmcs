'''
Created on 13.06.2019

@author: hspartali
'''

import numpy as np


def prepend_zero(array):
    return np.concatenate((np.zeros((1)), array))


def skip_extra_displacement():
    temp = 0

# testing
# force = np.array(pd.read_csv('C:\\Users\\hspartali\\Desktop\\CT80-35.csv', delimiter=';', decimal=',', skiprows=4, usecols=[1]))
# disp = np.array(pd.read_csv('C:\\Users\\hspartali\\Desktop\\CT80-35.csv', delimiter=';', decimal=',', skiprows=4, usecols=[3]))
# skip_noise_of_ascending_branch(force, disp)

# ar = np.array((5, 6, 7, 8, 9))
# 
# print(prepend_zero(ar))
