'''
Created on 13.06.2019

@author: hspartali
'''

import numpy as np


def prepend_zero(array):
    return np.concatenate((np.zeros((1)), array))


def skip_noise_of_ascending_branch_force(force, initial_force):
       
    indices = np.where(np.abs(force) > initial_force)[0]
    initial_force_index = indices[0]
    
    remaining_force = force[initial_force_index:]
    
    f_diff = np.abs(remaining_force[1:]) - np.abs(remaining_force[:-1])
    g_diff = np.array(f_diff[1:] * f_diff[:-1])
    idx1 = np.where((g_diff) <= 0)
    idx1 = idx1[0] + 1
    
    idx2 = idx1[1:] - idx1[0:-1]
    idx3 = np.where(np.abs(idx2) < 1)
    idx1 = list(idx1)
    
    for index in sorted(idx3[0], reverse=True):
        del idx1[np.int(index)]
    
    remaining_force = remaining_force[idx1].reshape(-1)
    
    force_filtered = np.concatenate((np.zeros((1)), remaining_force))    
    return force_filtered, initial_force_index, idx1


def skip_noise_of_ascending_branch_disp(disp, initial_force_index, idx1):
    remaining_disp = disp[initial_force_index:]
    remaining_disp = remaining_disp[idx1].reshape(-1)
    disp_filtered = np.concatenate((np.zeros((1)), remaining_disp))
    return disp_filtered


def skip_extra_displacement():
    temp = 0

# testing
# force = np.array(pd.read_csv('C:\\Users\\hspartali\\Desktop\\CT80-35.csv', delimiter=';', decimal=',', skiprows=4, usecols=[1]))
# disp = np.array(pd.read_csv('C:\\Users\\hspartali\\Desktop\\CT80-35.csv', delimiter=';', decimal=',', skiprows=4, usecols=[3]))
# skip_noise_of_ascending_branch(force, disp)

# ar = np.array((5, 6, 7, 8, 9))
# 
# print(prepend_zero(ar))
